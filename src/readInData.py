import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn
import sys, os
from sklearn import preprocessing

#import pandas_profiling as pandas_profiling
plt.style.use('seaborn-whitegrid')


def readInTaxi():
    """
    read in and clean dataset 
    return: df_train, df_test [pd.DataFrame]
    """
    df = pd.read_csv('../NYC_taxi/train_small.csv', nrows=50000, 
                           parse_dates=["pickup_datetime"])
    
    
    print('read in dataset -- training: ', df.shape)
       
    ## ==== step 1: missing values==================================
    df.dropna(how='any', axis='rows', inplace=True)

    
    
    ## === step 2: negative fareamount, constrained coordinates
    mask = (df['fare_amount'] > 0)
    df = df[mask]
    
    boxes = {'longitude': (-75, -73), 'latitude': (40, 42)}
    mask = (df['pickup_longitude'] >= boxes['longitude'][0])  & \
        (df['pickup_longitude'] <= boxes['longitude'][1]) & \
        (df['dropoff_longitude'] >= boxes['longitude'][0]) & \
        (df['dropoff_longitude'] <= boxes['longitude'][1]) & \
        (df['pickup_latitude'] >= boxes['latitude'][0]) & \
        (df['pickup_latitude'] <= boxes['latitude'][1]) & \
        (df['dropoff_latitude'] >= boxes['latitude'][0]) & \
        (df['dropoff_latitude'] <= boxes['latitude'][1])

    df = df[mask]
    print('*'*77)
    print('after drop dummies -- training: ', df.shape)

    return df


def distance(row):
    lat1 = row['dropoff_latitude']
    lat2 = row['pickup_latitude']
    lon1 = row['dropoff_longitude']
    lon2 = row['pickup_longitude']
    
    ## haversine formula https://en.wikipedia.org/wiki/Haversine_formula
    R = 6371
    dLat = np.pi/180 * (lat2 - lat1)
    dLon = np.pi/180 * (lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.pi/180*lat1) * np.cos(np.pi/180*lat2) * \
                + np.sin(dLon/2)**2
    c = 2 * np.arctan(np.sqrt(a)/np.sqrt(1-a))
    return R * c

def distance_to_poi(row, poi):
    lat1 = row['pickup_latitude']
    lon1 = row['pickup_longitude']
    lon2 = poi[0]
    lat2 = poi[1]
    R = 6371
    dLat = np.pi/180 * (lat2 - lat1)
    dLon = np.pi/180 * (lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.pi/180*lat1) * np.cos(np.pi/180*lat2) * \
                + np.sin(dLon/2)**2
    c = 2 * np.arctan(np.sqrt(a)/np.sqrt(1-a))
    return R * c

def feature_engineer(df):
    ## ==== step 1: calculate the point distance from pickup to dropoff=============
    df['distance'] = df.apply(distance, axis=1)
    
    ## ==== step 2: calculate the point distance to landmarks=============
    poi = {'nyc': (-74.006389, 40.714167),
       'jfk': (-73.782223, 40.644167),
       'ewr': (-74.175,  40.689722),
       'lga': (-73.87194, 40.774722)}
    
    
    for i in poi.keys():
        df['dist_to_{}'.format(i)] = df.apply(lambda x: distance_to_poi(x, poi[i]), axis=1)

    ## ==== step 3: clean the timestamp
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    
    features = ['fare_amount',  'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'hour', 'day_of_week', 'month', 'year', 'distance',
       'dist_to_nyc', 'dist_to_jfk', 'dist_to_ewr', 'dist_to_lga']
    
    print('*'*77)
    print('after feature engineered -- training: ', df.shape)
    
    return df[features]
