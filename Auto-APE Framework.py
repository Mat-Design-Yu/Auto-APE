# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:26:18 2023
The data set is automatically switched to realize the automatic calculation of model evaluation and model recommendation.
Input: A.csv file with labels in the first column and descriptors in the rest.
Output A.csv file with Each model, and the parameters selected by the model, as well as the evaluation indicators and rankings of each model.
@author: Administrator
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #Assign a training and test set
from sklearn.model_selection import GridSearchCV    #Grid search
from sklearn.model_selection import cross_val_score #Cross validation
import matplotlib.pyplot as plt   #Drawing function
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score


from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor as ADBR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor as CTBR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from lightgbm import LGBMRegressor as LGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn import svm
from xgboost import XGBRegressor as XGBR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.neighbors import KNeighborsRegressor as KNN

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import BayesSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_log_error
import pickle

#Global parameters
random_state_forall = 420
num_of_models = 15
n_iter_forall = 100
name_of_models = ['lr','ridge','adb','bpnn','cbr',
                  'etr','lgb','rfr','svr','xgb',
                  'lasso','dtr','gbr','br']


def Core(data):
    data = data.iloc[:,1:]
    data.drop_duplicates(inplace=True)  #Eliminate duplicate data
    data = data.reset_index(drop=True)
    x = data.iloc[:,1:]
    y = data.iloc[:,0]


    #The features are normalized to avoid the influence of data magnitude on the results, 
    #and the target y is not normalized here
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    # max_y = y.max()
    # min_y = y.min()
    # y = (y - min_y) / (max_y - min_y)
    
    #Delete the Nan column, directly delete all Nan columns, if not delete, 
    #the correlation coefficient screening model will not pass.
    x.dropna (axis=1, how='all', inplace=True) 
    x.fillna(0, inplace=True) #Replace the remaining Nans with zeros

    ###70% training set and 30% test set######
    from sklearn.model_selection import train_test_split
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.3,random_state=random_state_forall)
    
    def float_range(start, stop, step):
        ''' Support for float stepper functions
    
             Input:
                start (float)  : The count starts at start. The default is to start at 0.
                end   (float)  : The count ends up to stop, but does not include stop.
                step (float)  : Step size, default 1, for floating-point numbers, refer to steps for decimal places.
    
             Output:
                浮点数列表
    
             Example:
                >>> print(float_range(3.612, 5.78, 0.22))
                [3.612, 3.832, 4.052, 4.272, 4.492, 4.712, 4.932, 5.152, 5.372]
        '''
        start_digit = len(str(start))-1-str(start).index(".") # Take the number of decimal places of the start parameter
        stop_digit = len(str(stop))-1-str(stop).index(".")    # Take the number of decimal places of the end parameter
        step_digit = len(str(step))-1-str(step).index(".")    # Take the decimal number of the step parameter
        digit = max(start_digit, stop_digit, step_digit)      # Take the maximum number of decimal places
        return [(start*10**digit+i*step*10**digit)/10**digit for i in range(int((stop-start)//step))]
    
    #Model initialization
    lr = LR()
    ridge = Ridge()
    svr = svm.SVR()
    adb = ADBR()
    bpnn = MLPRegressor()
    cbr = CTBR()
    etr = ETR()
    lgb = LGBR()
    rfr = RFR()
    xgb = XGBR()
    lasso = Lasso()
    dtr = DTR()
    gbr = GBR()
    br = BR()
    
    
    #The selected model and parameter selection, the selection of parameters after multiple data verification,
    #while improving efficiency to ensure that the model can achieve the optimal.
    ###
    lr_grid_param = { 
                    'positive': [True, False],
    
                  }
    lr_Bayes = BayesSearchCV(lr , lr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    lr_Bayes.fit(Xtrain,Ytrain)
    
    ###
    ridge_grid_param = { 'alpha' : [1,0.1,0.01,0.001,0.0001,0],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                  }
    ridge_Bayes = BayesSearchCV(ridge , ridge_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    ridge_Bayes.fit(Xtrain,Ytrain)
    
    ###
    adb_grid_param = {  'learning_rate': [0.001, 0.01, 0.1, 1],
                        'loss': ['linear', 'square', 'exponential'],
                        'n_estimators': list(range(50, 150,30)),
                  }
    adb_Bayes = BayesSearchCV(adb , adb_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    adb_Bayes.fit(Xtrain,Ytrain)
    
    ###
    mytuple =[]
    j = 10
    for j in range(30,80,10):
        for i in range(5,10):
            ituple = (j,i)
            mytuple.append(ituple)
    bpnn_grid_param = { "hidden_layer_sizes": mytuple,
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'solver': ['lbfgs', 'sgd', 'adam'],
                        'alpha': [0.0001, 0.001, 0.01, 0.1],
                        'max_iter': [20, 40, 60],
                  }
    bpnn_Grid = GridSearchCV(bpnn , bpnn_grid_param  , n_jobs=6)
    bpnn_Grid.fit(Xtrain,Ytrain)
    ###
    cbr_grid_param = { 'iterations' : list(range(1000,4000,500)), 
                  'learning_rate' : float_range(0.01,0.15,0.01),
                  'depth' : list(range(1,10)),
                  }
    # cbr_Bayes = BayesSearchCV(cbr , cbr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    # cbr_Bayes.fit(Xtrain,Ytrain)
    cbr_Grid = GridSearchCV(cbr , cbr_grid_param  , n_jobs=6)
    cbr_Grid.fit(Xtrain,Ytrain)
    ###
    etr_grid_param = { 'n_jobs' : list(range(-5,-1)) , 
                  'n_estimators' : list(range(1,100,20)),
                  'max_depth' : list(range(10,30,5)),
                  }
    etr_Bayes = BayesSearchCV(etr , etr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    etr_Bayes.fit(Xtrain,Ytrain)
    ###
    lgb_grid_param = { 'n_estimators' : list(range(50,800,150)),
                  'learning_rate' : float_range(0.01,0.15,0.02),
                  'max_depth' : list(range(1,20,6)),
                  'num_leaves' : list(range(2,10,3))
                  }
    lgb_Bayes = BayesSearchCV(lgb , lgb_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    lgb_Bayes.fit(Xtrain,Ytrain)
    ###
    rfr_grid_param = { 'n_estimators' : list(range(50,800,150)),
                      'max_depth' : list(range(10,30,4)),
                      'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                  }
    rfr_Bayes = BayesSearchCV(rfr , rfr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    rfr_Bayes.fit(Xtrain,Ytrain)
    ###
    svr_grid_param = { 'C' : float_range(0.1,2.0,0.3) , 
                  'kernel' : ['rbf','poly','sigmoid'],
                  'gamma' : list(range(1, 5)),
                  'epsilon': list(float_range(0.01, 0.1, 0.01)), 
                    'cache_size': list(range(2000, 5000, 1000)), 
                  }
    svr_Bayes = BayesSearchCV(svr , svr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    svr_Bayes.fit(Xtrain,Ytrain)
    ###
    xgb_grid_param = { 'n_estimators' : list(range(50,800,150)),
                  'learning_rate' : float_range(0.01,0.15,0.02),
                  'subsample' : float_range(0.45,0.60,0.02),
                  }
    xgb_Bayes = BayesSearchCV(xgb , xgb_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    xgb_Bayes.fit(Xtrain,Ytrain)
    ###
    lasso_grid_param = { 'alpha': float_range(0.1, 1.0, 0.1),
                        'max_iter': list(range(1000, 10001, 1000))
                  }
    lasso_Bayes = BayesSearchCV(lasso , lasso_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    lasso_Bayes.fit(Xtrain,Ytrain)
    ###
    dtr_grid_param = { 'max_depth': list(range(5, 30, 5)),
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                  }
    dtr_Bayes = BayesSearchCV(dtr , dtr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    dtr_Bayes.fit(Xtrain,Ytrain)
    ###
    gbr_grid_param = { 'n_estimators': list(range(50, 800, 150)),
                        'learning_rate': float_range(0.01, 0.15, 0.02),
                        'max_depth': list(range(3, 10, 2)),
                        'subsample': float_range(0.6, 1.0, 0.1),
                  }
    gbr_Bayes = BayesSearchCV(gbr , gbr_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    gbr_Bayes.fit(Xtrain,Ytrain)
    ###
    br_grid_param = { 'n_estimators': list(range(50, 800, 150)),
                        'max_samples': float_range(0.1, 1.0, 0.2),
                        'max_features': float_range(0.1, 1.0, 0.2),
                  }
    br_Bayes = BayesSearchCV(br , br_grid_param , n_iter=n_iter_forall , random_state = random_state_forall)
    br_Bayes.fit(Xtrain,Ytrain)
    
    print("Parameter selection is completed")
    #####Modeling#######
    lr = LR(
            **lr_Bayes.best_params_
        ).fit(Xtrain, Ytrain)#Linear regression
    ridge = Ridge(
        **ridge_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Ridge regression
    
    adb = ADBR(
        **adb_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#AdaBoost
    
    bpnn = MLPRegressor(
        **bpnn_Grid.best_params_
                        ).fit(Xtrain,Ytrain)#Multilayer perceptron
    cbr = CTBR(
        # **cbr_Bayes.best_params_
        **cbr_Grid.best_params_
               ).fit(Xtrain,Ytrain)#CatBoost
    etr = ETR(
        **etr_Bayes.best_params_
              ).fit(Xtrain,Ytrain)#Extremely random tree
    lgb = LGBR( 
        **lgb_Bayes.best_params_
               ).fit(Xtrain,Ytrain)#LightGBM
    rfr = RFR(
        **rfr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Random forest
    svr = svm.SVR(
        **svr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Support vector machine
    xgb = XGBR(
        **xgb_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#XGBoost
    lasso = Lasso(
        **lasso_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Lasso
    dtr = DTR(
        **dtr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Decision tree regression
    gbr = GBR(
        **gbr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Gradient boosting
    br = BR(
        **br_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Bagging
    
    
    
    models=[]
    models.append(lr)
    models.append(ridge)
    models.append(adb)
    models.append(bpnn)
    models.append(cbr)
    models.append(etr)
    models.append(lgb)
    models.append(rfr)
    models.append(svr)
    models.append(xgb)
    models.append(lasso)
    models.append(dtr)
    models.append(gbr)
    models.append(br)
    
    
    #Forecasting
    yhat_lr = lr.predict(Xtest)
    yhat_ridge = ridge.predict(Xtest)
    yhat_adb = adb.predict(Xtest)
    yhat_bpnn = bpnn.predict(Xtest)
    yhat_cbr = cbr.predict(Xtest)
    yhat_etr = etr.predict(Xtest)
    yhat_lgb = lgb.predict(Xtest)
    yhat_rfr = rfr.predict(Xtest)
    yhat_svr = svr.predict(Xtest)
    yhat_xgb = xgb.predict(Xtest)
    yhat_lasso = lasso.predict(Xtest)
    yhat_dtr = dtr.predict(Xtest)
    yhat_gbr = gbr.predict(Xtest)
    yhat_br = br.predict(Xtest)
    
    yhat_list=[]
    for i in range(len(models)):
        yhat_list.append(models[i].predict(Xtest))
    
        
    
    model_pass_check_list = models
    
    
    evaluations = ['R2','RMSE','MSLE','MEDAE','MAE','EVS'
                    ,'10kf_R2','10kf_RMSE','10kf_MSLE','10kf_MEDAE','10kf_MAE','10kf_EVS'
                   ]
    acc_data = []
    for i in range(len(model_pass_check_list)):
        #Model accuracy
        model_evaluations = []
        model_evaluations.append(model_pass_check_list[i].score(Xtest,Ytest))
        model_evaluations.append(mean_squared_error(Ytest,model_pass_check_list[i].predict(Xtest))**0.5)
        model_evaluations.append(mean_squared_log_error(abs(Ytest),abs(model_pass_check_list[i].predict(Xtest))))
        model_evaluations.append(median_absolute_error(Ytest,model_pass_check_list[i].predict(Xtest)))
        model_evaluations.append(mean_absolute_error(Ytest,model_pass_check_list[i].predict(Xtest)))
        model_evaluations.append(explained_variance_score(Ytest,model_pass_check_list[i].predict(Xtest))) 
        #Model stability
        model_evaluations.append(cross_val_score(model_pass_check_list[i],x,y,cv=10).mean())
        model_evaluations.append(-(-(cross_val_score(model_pass_check_list[i],x,y,cv=10
                                                  ,scoring = "neg_mean_squared_error").mean()))**0.5)
        #----
        y_pred = cross_val_predict(model_pass_check_list[i], x, y, cv=10)                               
        model_evaluations.append(mean_squared_log_error(abs(y), abs(y_pred)))
        #----
        model_evaluations.append(cross_val_score(model_pass_check_list[i],x,y,cv=10
                                                  ,scoring = "neg_median_absolute_error").mean())
        model_evaluations.append(cross_val_score(model_pass_check_list[i],x,y,cv=10
                                                  ,scoring = "neg_mean_absolute_error").mean())
        model_evaluations.append(cross_val_score(model_pass_check_list[i],x,y,cv=10
                                                  ,scoring = "explained_variance").mean())
        #Import the evaluation criteria matrix
        acc_data.append(model_evaluations)
    #Convert to DataFrame format
    acc_df = pd.DataFrame(acc_data,index=model_pass_check_list,columns=evaluations,dtype=float)


    #Model ranking
    sort_name_list = ["R2_sort",'RMSE_sort','MSLE_sort','MEDAE_sort','MAE_sort','EVS_sort'
                       ,'10kf_R2_sort','10kf_RMSE_sort','10kf_MSLE_sort','10kf_MEDAE_sort','10kf_MAE_sort','10kf_EVS_sort'
                      ]
    for i in range(len(evaluations)):
        acc_df.sort_values(by=evaluations[i], inplace=True, ascending=True)
        sort_index = list(range(len(model_pass_check_list)))
        if sort_name_list[i]=="RMSE_sort" or sort_name_list[i]=="MSLE_sort"or sort_name_list[i]=="MEDAE_sort" or sort_name_list[i]=="MAE_sort" or sort_name_list[i]=="10kf_MSLE_sort":
            acc_df[sort_name_list[i]]=sort_index
        else:    
            sort_index.reverse()
            acc_df[sort_name_list[i]]=sort_index
    #Total sorting
    acc_df['sort_score'] = acc_df.iloc[:,18:].sum(axis=1)
    #The better the model, the better the model
    acc_df.sort_values(by='sort_score', inplace=True, ascending=True)
    
    return acc_df,models




import pandas as pd
import os
import joblib
import shutil

# Let's say your dataset folder name is datasets
dataset_folder = "datasets"
output_folder = "results"  # The name of the folder to use to save the results

# Check if the results folder exists and create it if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

name_of_models = ['lr','ridge','adb','bpnn','cbr',
                  'etr','lgb','rfr','svr','xgb',
                  'lasso','dtr','gbr','br']

# Iterate over the files in the dataset folder
for filename in os.listdir(dataset_folder):
    if filename.endswith(".csv"):
        dataset_name = os.path.splitext(filename)[0]  # Extract the filename (without the extension) as the dataset name

        dataset_path = os.path.join(dataset_folder, filename)  # Build the full path to the dataset file
        output_dataset_folder = os.path.join(output_folder, dataset_name)  # Build the full path to the results folder

        # Check if the results folder exists and create it if not
        if not os.path.exists(output_dataset_folder):
            os.makedirs(output_dataset_folder)

        data = pd.read_csv(dataset_path)  # Read the dataset as a DataFrame
        # Here the core() function is called to perform the calculation and save the results to the results folder
        result = Core(data)
        acc_df, models = result

        # Save the evaluation metrics as CSV files
        acc_csv_filename = os.path.join(output_dataset_folder, "modelAcc.csv")
        acc_df.to_csv(acc_csv_filename)

        save_directory = os.path.join(output_dataset_folder, "models")  # The name of the folder where the model is saved
        # Check if the directory exists and create it if not
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # The list of models is traversed and saved as a pickle file
        for name, model in zip(name_of_models,models):
            model_name = f'model_{name}.pickle'
            model_path = os.path.join(save_directory, model_name)
            
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            print(f"保存模型 {model_name}")
        
        print("模型保存完成。")

        # Copy the metrics CSV file to the models folder
        acc_csv_destination = os.path.join(save_directory, "modelAcc.csv")
        shutil.copyfile(acc_csv_filename, acc_csv_destination)
        print(f"复制评价指标文件到 {acc_csv_destination} 完成")

        # Save the test dataset as a CSV file
        data_filename = os.path.join(output_dataset_folder, filename)
        data.to_csv(data_filename)