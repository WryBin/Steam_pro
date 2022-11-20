import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

import statsmodels.tsa.arima.model as arima

from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer


def linear_regression(train_data, val_data, pre_data, features):

    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    model = model.fit(train_data[features], train_data['Radiation'])

    # get information about importance of features
    'intercept:'+ str(model[1].intercept_)
    sorted(dict(zip(features, model[1].coef_)).items(), key=lambda x:x[1], reverse=True)

    Y_train = model.predict(train_data[features])
    Y_val = model.predict(val_data[features])
    Y_pred = model.predict(pre_data[features])
    return Y_train, Y_val, Y_pred, model

def linear_regression_cv(train_data, features):

    train_data.dropna(inplace=True)

    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    scores = cross_val_score(model, train_data[features], train_data['Radiation'], cv=5, scoring=make_scorer(mean_absolute_error))

    return model, scores
    

def LightGBM(train_data, val_data, pred_data, features, Mode):

    params = {
    'boosting': 'gbdt',
    'objective': 'rmse',
    'num_leaves': 300,
    'learning_rate': 0.1,
    'metric': {'rmse'},
    'verbose': -1,
    'min_data_in_leaf': 6,
    'max_depth':30,
    'seed':42, 
    }

    # lgb_train = lgb.Dataset(train_data[features], train_data['Steam_flow'].values, weight=sample_weight[:len(train_data)])
    lgb_train = lgb.Dataset(train_data[features], train_data['Steam_flow'].values)
    lgb_eval = lgb.Dataset(val_data[features], val_data['Steam_flow'].values, reference=lgb_train)

    gbm = lgb.train(params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=300, 
                    callbacks=[lgb.early_stopping(stopping_rounds=100)],
                    )

    Y_train = gbm.predict(train_data[features], num_iteration=gbm.best_iteration)
    Y_val = gbm.predict(val_data[features], num_iteration=gbm.best_iteration)
    Y_pred = gbm.predict(pred_data[features], num_iteration=gbm.best_iteration)


    return Y_train, Y_val, Y_pred, gbm


def LightGBM_param_cv(train_data, val_data, pred_data, features, Mode):

    num_leaves = [280, 300, 320, 340, 400, 500]
    learning_rate = [0.1]
    max_depth = [28, 30, 32, 34, 36, 40, 50]
    min_data_in_leaf = [5, 6, 7]

    params = {
    'boosting': ['gbdt'],
    'objective': ['rmse'],
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'metric': ['rmse'],
    'verbose': [-1],
    'min_data_in_leaf': min_data_in_leaf,
    'max_depth': max_depth,
    'seed':[42], 
    # 'sub_feature': [0.7], 
    }

    clf = GridSearchCV(LGBMRegressor(), params, cv=5, scoring='neg_mean_squared_error')
    clf.fit(train_data[features], train_data['Steam_flow'].values)

    return clf


def XGBoost(train_data, val_data, pred_data, features, Mode):

    dtrain = xgb.DMatrix(train_data[features], label=train_data['Radiation'].values)
    dval = xgb.DMatrix(val_data[features], label=val_data['Radiation'].values)
    dpred = xgb.DMatrix(pred_data[features], label=pred_data['Radiation'].values)

    # specify parameters via map
    param = {'max_depth':7, 'eta':0.1, 'objective':'reg:squarederror', 'min_child_weight': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'lambda': 0.1}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)

    # make prediction
    Y_train = bst.predict(dtrain)
    Y_val = bst.predict(dval)
    Y_pred = bst.predict(dpred)


    return Y_train, Y_val, Y_pred, bst


def XGBoost_param_cv(train_data, val_data, pred_data, features, Mode):

    # dtrain = xgb.DMatrix(train_data[features], label=train_data['Radiation'].values)
    # dval = xgb.DMatrix(val_data[features], label=val_data['Radiation'].values)
    # dpred = xgb.DMatrix(pred_data[features], label=pred_data['Radiation'].values)

    max_depth = [7]
    eta = [0.1]
    min_child_weight = [3, 4, 5]


    # specify parameters via map
    param = {'max_depth':max_depth, 'eta':eta, 'objective':['reg:squarederror'], 'min_child_weight': min_child_weight, 'subsample': [0.8],\
            'colsample_bytree': [0.8], 'gamma': [0.1], 'lambda': [0.1]}
    num_round = 100
    clf = GridSearchCV(XGBRegressor(), param, cv=5, scoring='neg_mean_squared_error')
    clf.fit(train_data[features], train_data['Radiation'].values)

    return clf

def Prophet(train_data, val_data, pred_data, features, Mode):
    pass

def Arima(train_data, val_data, pred_data, features, Mode):
    
    result = arima.ARIMA(train_data['Radiation'].values, order=(1, 1, 1)).fit().summary()

    return result

def ElasticNet(train_data, val_data, pred_data, features, Mode):
    
    train_data = train_data.dropna()
    val_data = val_data.dropna()
    pred_data = pred_data.dropna()

    clf = ElasticNet(alpha=0.1, l1_ratio=0.1, random_state=42)
    clf.fit(train_data[features], train_data['Radiation'].values)

    Y_train = clf.predict(train_data[features])
    Y_val = clf.predict(val_data[features])
    Y_pred = clf.predict(pred_data[features])

    return Y_train, Y_val, Y_pred, clf

def ElascicNet_cv(train_data, val_data, pred_data, features, Mode):

    train_data = train_data.dropna()
    param = {'alpha': [0.1, 0.5, 1], 'l1_ratio': [0.1, 0.5, 1]}
    clf = GridSearchCV(ElasticNet(), param, cv=5, scoring='neg_mean_squared_error')
    clf.fit(train_data[features], train_data['Radiation'].values)

    return clf



