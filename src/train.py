from sklearn.model_selection import train_test_split,  KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization

from readInData import *

near model
def model_linear(x_train, y_train, x_test, y_test):   
    performance = pd.DataFrame(columns=['train_RMSE', 'test_RMSE'])
    features = ['distance', 'passenger_count', 'hour', 'day_of_week', 'month', \
           'dist_to_nyc', 'dist_to_jfk', 'dist_to_ewr', 'dist_to_lga']
    
    lr = LinearRegression()

    lr.fit(x_train[features], y_train)
       
    idx = 0
    y_pred = lr.predict(x_train[features])
    performance.loc[idx, 'train_RMSE'] = np.sqrt(mean_squared_error(y_train, y_pred))
        
    y_pred = lr.predict(x_test[features])
    performance.loc[idx, 'test_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
    return performance
    

def model_lgbm(x_train, y_train, x_test, y_test, **params):
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])

    features = ['distance', 'passenger_count', 'hour', 'day_of_week', 'month', \
           'dist_to_nyc', 'dist_to_jfk', 'dist_to_ewr', 'dist_to_lga']
    x_train = x_train[features]
    x_test = x_test[features]
    
    
    # start training
    lgbm = LGBMRegressor(nthread=-1, silent=-1, verbose=-1,
                        **params)
    lgbm.fit(x_train, y_train, 
             eval_set=[(x_train, y_train), (x_test, y_test)],
             eval_metric = 'rmse', 
             verbose=False, 
             early_stopping_rounds=200)
    
    
    performance = pd.DataFrame(columns=['train_RMSE', 'test_RMSE'])
    idx = 0
    y_pred = lgbm.predict(x_train, num_iteration=lgbm.best_iteration_)
    performance.loc[idx, 'train_RMSE'] = np.sqrt(mean_squared_error(y_train, y_pred))
        
    y_pred = lgbm.predict(x_test, num_iteration=lgbm.best_iteration_)
    performance.loc[idx, 'test_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
   
    
    return performance


def model_evaluate(**params):
    performance = model_lgbm(x_train, y_train, x_test, y_test, **params)
    return - performance['test_RMSE'].iloc[0]
  

def main():
    params =  {'num_leaves': (2, 50), 
          'max_depth': (2, 9),
          'min_split_gain': (.01, .03),
          'learning_rate': (.01, .02), 
          'reg_alpha': (.03, .05), 
          'reg_lambda': (.06, .08),             
          'colsample_bytree': (0.8, 1),       
          'subsample': (0.8, 1), 
          'min_child_weight': (38, 40)}

    df = readInTaxi()
    df = feature_engineer(df)
    _train, x_test, y_train, y_test = train_test_split(df.drop('fare_amount', axis=1),
                                                   df['fare_amount'], 
                                                   test_size = 0.3,
                                                   random_state = 42)


    ## train the model
    bo = BayesianOptimization(model_evaluate, params)
    bo.maximize(init_points=5, n_iter=5)
    best_params = bo.res['max']['max_params']
    best_performance = model_lgbm(x_train, y_train, x_test, y_test, **best_params)

    print('best hyper parameter set: ', best_params)    
    print('lightGBM RMSE score: ', best_performance)
    return best_performance
    

if __name__ == '__main__':
    main()
