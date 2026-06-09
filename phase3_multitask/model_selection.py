# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:43:05 2021

@author: win
"""
# 正则化：降低模型的复杂度，避免过拟合。

# 加载模块
from sklearn.datasets import load_iris
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.mixture import GaussianMixture
from sklearn import utils, kernel_ridge, gaussian_process, ensemble, linear_model, neighbors, preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, Lasso, ElasticNet, Perceptron
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import pickle
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt


# 分割数据集
def load_csv_name(path=r'C:\Users\ww\Desktop', dataname=r'jqxx.txt', skiprow=2):
    try:
        if path is not None:
            os.chdir(path)
        else:
            path = os.getcwd()
        open(dataname)
    except IOError:
        raise IOError('No file:{} in {} '.format(dataname, path))

    data = pd.read_csv(dataname)
    return data


def model_selection(X_train, Y_train, test_x, test_y, model_name, scoring, rs):
    best_param = None
    if model_name == "KNR":
        model = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                              metric='minkowski')  # n_neighbors是设定邻居的个数
        n_neighbors = [2, 3, 4, 5, 6, 7]
        param_grid = dict(n_neighbors=n_neighbors)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "SVR":
        model = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=False, max_iter=1000000000)
        kernel = ['rbf']  # ,'rbf''linear'
        C = [21500, 20000, 15000, 10000, 7500, 6000, 5000, 4000, 3000, 2000, 1000, 500]
        gamma = [0.001, 0.01, 0.02, 0.03, 0.031, 0.04, 0.05, 0.06, 0.09, 0.1, 0.15, 0.2]
        # C = [5000]
        # gamma = [0.01]
        param_grid = dict(kernel=kernel, C=C, gamma=gamma)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        grid_result = grid_search.fit(X_train, Y_train)
        y_predict = grid_search.predict(test_x)
        mse = mean_squared_error(test_y, y_predict)
        print("Best: %f using %s" % (mse, grid_search.best_params_))
        # print(y_predict)
        # path_m = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\ago\model\GBR" + str(rs) + ".model"
        # with open(path_m, "wb") as f:
        #     pickle.dump(grid_search, f)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_

        print("Best grid: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "BayesianRidge":
        model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True,
                              lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.01, verbose=False)
        alpha_1 = [1e-08, 1e-07, 1e-06, 1e-05]
        alpha_2 = [1e-07, 1e-05, 1e-03]
        param_grid = dict(alpha_1=alpha_1, alpha_2=alpha_2)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "SGDRL2":
        model = SGDRegressor(alpha=0.0001, average=False,
                             epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                             learning_rate='invscaling', loss='squared_loss', max_iter=1000000,
                             n_iter_no_change=5, penalty='l2', power_t=0.25,
                             random_state=0, shuffle=True, tol=0.01,
                             verbose=0, warm_start=False)
        alpha = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "kernelridge":
        kernel = 1.0 * RBF(1.0)
        model = kernel_ridge.KernelRidge(alpha=1, kernel=kernel, gamma="scale", degree=3, coef0=1, kernel_params=None)
        alpha = [100, 10, 1, 0.1, 0.01, 0.001]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "logisticregressor":
        model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                   class_weight=None, random_state=None, solver='liblinear', max_iter=1000000,
                                   multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
        penalty = ['l1', 'l2']
        C = [100, 10, 5, 1, 0.1]
        param_grid = dict(penalty=penalty, C=C)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        grid_result = grid_search.fit(X_train, Y_train)
        # with open(r"C:\Users\win\Desktop\logr.model", "wb") as f:
        #     pickle.dump(grid_search, f)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "GPR":
        model = gaussian_process.GaussianProcessRegressor(alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                                          n_restarts_optimizer=0,
                                                          normalize_y=False, copy_X_train=True, random_state=0)
        alpha = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "RFR":
        model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                               min_impurity_decrease=0.0,
                                               bootstrap=True, oob_score=False,
                                               random_state=None, verbose=0, warm_start=False)
        n_estimators = [70, 80, 90, 100]
        max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
        # max_depth = [10]
        # n_estimators = [90]
        param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
        # path_m = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\ago\RFR_model\RFR" + str(rs) + ".model"
        # with open(path_m, "wb") as f:
        #     pickle.dump(grid_search, f)

    if model_name == "GBR":
        model = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                                   subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                                   min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                                   max_depth=3, min_impurity_decrease=0.,
                                                   init=None, random_state=None,
                                                   max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                                   warm_start=False)
        max_depth = [1, 2, 3, 4, 5, 6, 7]

        # max_depth = [5]
        param_grid = dict(max_depth=max_depth)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        grid_result = grid_search.fit(X_train, Y_train)
        y_predict = grid_search.predict(test_x)
        mse = mean_squared_error(test_y, y_predict)
        # print("Best: %f using %s" % (mse,grid_search.best_params_))
        # print(y_predict)
        # path_m = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\ago\GBR_model\GBR"+str(rs)+".model"
        # with open(path_m, "wb") as f:
        #     pickle.dump(grid_search, f)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best grid: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "AdaBR":
        model = AdaBoostRegressor(n_estimators=100, learning_rate=1.,
                                  random_state=0)
        n_estimators = [50, 100, 200, 300, 400, 500]
        learning_rate = [0.7, 0.8, 0.9, 1, 1.1]
        param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "TreeR":
        model = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features=None, random_state=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None)
        max_depth = [3, 4, 5, 6, 7]
        param_grid = dict(max_depth=max_depth)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(6, 2), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        alpha = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "ElasticNet":
        model = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False,
                           max_iter=100000,
                           copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)
        alpha = [0.0001, 0.001, 0.01, 0.1, 1]
        l1_ratio = [0.3, 0.5, 0.8]
        param_grid = dict(alpha=alpha, l1_ratio=l1_ratio)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "Lasso":
        model = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=100000,
                      tol=0.001, warm_start=False, positive=False, random_state=None, )
        alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    if model_name == "SGDRL1":
        model = SGDRegressor(alpha=0.0001, average=False,
                             epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                             learning_rate='invscaling', loss='squared_loss', max_iter=1000000,
                             n_iter_no_change=5, penalty='l1', power_t=0.25,
                             random_state=0, shuffle=True, tol=0.01,
                             verbose=0, warm_start=False)
        alpha = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
        param_grid = dict(alpha=alpha)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
        print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    if model_name == "xgboost":
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, min_child_weight=1, seed=0, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1)
        n_estimators = [550, 575, 600, 650, 675]
        learning_rate = [0.01, 0.05, 0.07, 0.1, 0.2]
        subsample = [0.6, 0.7, 0.8, 0.9]
        colsample_bytree = [0.6, 0.7, 0.8, 0.9]
        gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
        min_child_weight = [1, 2, 3, 4, 5, 6]
        reg_alpha = [0.05, 0.1, 1, 2, 3]
        reg_lambda = [0.05, 0.1, 1, 2, 3]
        param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                          colsample_bytree=colsample_bytree, gamma=gamma, max_depth=max_depth,
                          min_child_weight=min_child_weight, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        grid_result = grid_search.fit(X_train, Y_train)
        best_param = grid_search.best_params_
        best_score = grid_result.best_score_
    return best_param, best_score


def model_train(model_list, path, dataname, n_rs, scoring):
    outcome = {}
    data = load_csv_name(path, dataname, skiprow=1)
    data = data.values
    for model_name in model_list:
        all_score = 0
        print(model_name)
        # for rs in range(n_rs):
        X = data[:, 0:14]
        scaler = StandardScaler()
        data_X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data[:, 14], test_size=0.1, random_state=n_rs)

        best_param, best_score = model_selection(X_train, y_train, X_test, y_test, model_name=model_name,
                                                 scoring=scoring, rs=n_rs)
        all_score += best_score
        # out_score = all_score/n_rs
        # outcome[model_name] = out_score

    return outcome, X_train, X_test, y_train, y_test


import os
import pandas as pd
import numpy as np
import joblib
import openpyxl
from sklearn import preprocessing
import xlrd


def load_csv_name(path=r'C:\Users\ww\Desktop', dataname=r'jqxx.txt', skiprow=2):
    try:
        if path is not None:
            os.chdir(path)
        else:
            path = os.getcwd()
        open(dataname)
    except IOError:
        raise IOError('No file:{} in {} '.format(dataname, path))

    data = pd.read_csv(dataname)
    return data


out_error = dict()
for n_rs in range(0, 100):
    path = r'E:\其它工作\奥氏体肿胀数据库结题\奥氏体肿胀数据库结题'
    dataname = r'input_all.csv'
    model_list = ["RFR"]
    # 进行k次随机采样
    out_score, X_train, X_test, y_train, y_test = model_train(model_list, path, dataname, n_rs,
                                                              "neg_mean_squared_error")
    # pd.DataFrame(X_train).to_csv(r"E:\文本挖掘\工作三-工艺抽取\gama-ML\97_10\3_97_10\x_train.csv")
    # pd.DataFrame(X_test).to_csv(r"E:\文本挖掘\工作三-工艺抽取\gama-ML\97_10\3_97_10\X_test.csv")
    # pd.DataFrame(y_train).to_csv(r"E:\文本挖掘\工作三-工艺抽取\gama-ML\97_10\3_97_10\y_train.csv")
    # pd.DataFrame(y_test).to_csv(r"E:\文本挖掘\工作三-工艺抽取\gama-ML\97_10\3_97_10\ y_test.csv")
    # print(out_score)

    # model_list = ["KNR","kernelridge","SVR","BayesianRidge","SGDRL2","RFR","GBR","ElasticNet","Lasso","SGDRL1"]

    xls = openpyxl.Workbook()
    sht = xls.create_sheet(index=0)

    tain_data_path = r'E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add'
    train_dataname = r'input_all.csv'
    data = load_csv_name(tain_data_path, train_dataname, skiprow=1)
    data = data.values
    data_x = data[:, 0:14]
    scaler = preprocessing.StandardScaler().fit(data_x)

    compostion_path = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\2023articles"
    dataname = "two_feng.csv"
    data = load_csv_name(compostion_path, dataname, skiprow=0)
    data = data.values

    model_path = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\ago\RFR_model\RFR" + str(n_rs) + ".model"
    model = joblib.load(filename=model_path)
    row_len = np.size(data, 0)
    predict_out = list()
    for row in range(row_len):
        x = data[row, 0:14]
        shape_x = x.reshape(1, -1)
        scaler_x = scaler.transform(shape_x)
        predict_y = model.predict(scaler_x)
        predict_out.append(predict_y)
    y_path = r"E:\文本挖掘\工作三-工艺抽取\gama-ML\2_add\2023articles\true.xlsx"
    xls = xlrd.open_workbook(y_path)
    sht = xls.sheet_by_index(0)
    y_true = sht.col_values(0)
    # y_true = np.array(y_true)
    # predict_out = np.array(predict_out)
    all_re_error = 0
    for i in range(len(y_true)):
        pre_i = predict_out[i]
        true_i = y_true[i]
        relative_error = abs(pre_i - true_i) / true_i
        all_re_error += relative_error
    mean_relative_error = all_re_error / len(y_true)
    out_error[n_rs] = mean_relative_error