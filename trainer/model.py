#!/user/bin/env python 

import numpy as np
import shutil
import pandas as pd
import sklearn
import sys, os
from sklearn import preprocessing
import tensorflow as tf


from tensorflow_transform.saved import input_fn_maker, saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io


tf.logging.set_verbosity(tf.logging.INFO)



# List of CSV columns
CSV_COLUMNS = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', \
                'passengers', 'key']

# Choose which column is the label
LABEL_COLUMN = 'fare_amount'

# set the default values for each CSV columns in case there is a missing value 
DEFAULTS =  [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# create an input function that stores your data into a dataset 
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(list(zip(CSV_COLUMNS, columns)))
            label = features.pop(LABEL_COLUMN)
            return features, label 

        # create a list of files that match pattern
        file_list = tf.gfile.Glob(filename)
    
        # create dataset form file list 
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None 
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end of input 

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn 


def distance(lat1, lat2, lon1, lon2):
    ## haversine formula https://en.wikipedia.org/wiki/Haversine_formula
    R = 6371
    dLat = np.pi/180 * (lat2 - lat1)
    dLon = np.pi/180 * (lon2 - lon1)
    a = tf.math.sin(dLat/2)**2 + tf.math.cos(np.pi/180*lat1) * tf.math.cos(np.pi/180*lat2) * \
                + tf.math.sin(dLon/2)**2
    c = 2 * tf.math.atan2(tf.sqrt(a)/tf.sqrt(1-a))
    return R * c
 


def add_more_features(row):
    ## this is how to calculate the distance in euclidean distance
    lat1 = row['dropoff_latitude']
    lat2 = row['pickup_latitude']
    lon1 = row['dropoff_longitude']
    lon2 = row['pickup_longitude']
    
   row['distance'] = distance(lat1, lat2, lon1, lon2)

    ## this is how to add distance to point of interest
    poi = {'nyc': (-74.006389, 40.714167),
       'jfk': (-73.782223, 40.644167),
       'ewr': (-74.175,  40.689722),
       'lga': (-73.87194, 40.774722)}
   
    for i in poi.keys():
        row['dist_to_{}'.format(i)] = distance(row['pickup_latitude'], row['pickup_longitude'], poi[i][1], poi[i][0])

    return row
   


# define your feature columns 
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('passengers'),

    # define features
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),
    tf.feature_column.categorical_column_with_identity('day_of_week', num_buckets=100) # some large number
    tf.feature_column.categorical_column_with_identity('month', num_buckets=12),

    # engineered fratures that created in the input fn 
    tf.feature_column.numeric_column('distance'),
    tf.feature_column.numeric_column('dist_to_nyc'),
    tf.feature_column.numeric_column('dist_to_jfk'),
    tf.feature_column.numeric_column('dist_to_ewr'),
    tf.feature_column.numeric_column('dist_to_lga')]



### build the esimator 
def build_estimator(model_dir, nbuckets, hidden_unites):
    """
    build an estimator starting from INPUT_COLUMNS 
    wide and deep model
    """

    # Input columns 
    plon, plat, dlon, dlat, pcount, hour_of_day, day_of_week, month_of_year, distance, dist_to_nyc, dist_to_jfk, dist_to_ewr, dist_to_lga = INPUT_COLUMNS 

    # bucketize the lats and lons 
    latbuckets = np.linspace(40.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-75.0, -73.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    day_hr =  tf.feature_column.crossed_column([day_of_week, hour_of_day], 24 * 7)


    # wide and deep columns 
    wide_columns = [dloc, ploc, pd_pair, day_hr, day_of_week, hour_of_day, pcount]
    deep_columns = [tf.feature_columns.embedding_column(pd_pair, 10), 
                    tf.feature_columns.embedding_column(day_hr, 10),
                    plat, plon, dlat, dlon, latdiff, londiff, distance, dist_to_nyc, dist_to_jfk, dist_to_ewr, dist_to_lgad
    ]


    # set checkpoint 
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 30, 
                                        keep_checkpoint_max = 3)
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units,
        config = run_config)

    # add evaluation metrix 
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator



### create estiamtor train and evaluate function 
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = 100,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)




### evaludation matric 
def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
    }


                
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']

