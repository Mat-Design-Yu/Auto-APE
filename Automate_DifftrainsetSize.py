# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:26:18 2023
自动改变TrainSet Size,验证大多数数据是冗余数据
@author: 25304
"""
#%% 导入库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #分配训练集与测试集
from sklearn.model_selection import GridSearchCV    #网格搜索
from sklearn.model_selection import cross_val_score #交叉验证
import matplotlib.pyplot as plt   #画图
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
import math

#%% Auto-APE部分
#全局参数
random_state_forall = 420
num_of_models = 15
n_iter_forall = 100
name_of_models = ['lr','ridge','adb','bpnn','cbr',
                  'etr','lgb','rfr','svr','xgb',
                  'lasso','dtr','gbr','br']


def Core(data,Trainset_Size=0.3):
    data = data.iloc[:,1:]
    data.drop_duplicates(inplace=True)  #剔除重复的数据
    data = data.reset_index(drop=True)
    x = data.iloc[:,1:]
    y = data.iloc[:,0]


    #对特征进行归一化，避免数据量级对结果的影响，此处目标y不进行归一化
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    # max_y = y.max()
    # min_y = y.min()
    # y = (y - min_y) / (max_y - min_y)

    x.dropna (axis=1, how='all', inplace=True) #删除含Nan的列,直接删除全为Nan的列，如果不删，后面相关系数筛选模型会没有模型通过
    x.fillna(0, inplace=True) #将剩下的Nan替换为0，

    ###七三、八二分训练集和测试集######
    # from sklearn.model_selection import train_test_split
    # Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=Trainset_Size,random_state=random_state_forall)
    
    from sklearn.model_selection import train_test_split

    # 划分出临时集和验证集
    Xtemp, Xvalid, Ytemp, Yvalid = train_test_split(x, y, test_size=0.1, random_state=random_state_forall)
    
    # 再次划分出训练集和测试集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtemp, Ytemp, test_size=(1-Trainset_Size), random_state=random_state_forall)
    
    # 现在，Xtrain和Ytrain是训练集，Xvalid和Yvalid是验证集，Xtest和Ytest是测试集

    
    def float_range(start, stop, step):
        ''' 支持 float 的步进函数
    
            输入 Input:
                start (float)  : 计数从 start 开始。默认是从 0 开始。
                end   (float)  : 计数到 stop 结束，但不包括 stop。
                step (float)  : 步长，默认为 1，如为浮点数，参照 steps 小数位数。
    
            输出 Output:
                浮点数列表
    
            例子 Example:
                >>> print(float_range(3.612, 5.78, 0.22))
                [3.612, 3.832, 4.052, 4.272, 4.492, 4.712, 4.932, 5.152, 5.372]
        '''
        start_digit = len(str(start))-1-str(start).index(".") # 取开始参数小数位数
        stop_digit = len(str(stop))-1-str(stop).index(".")    # 取结束参数小数位数
        step_digit = len(str(step))-1-str(step).index(".")    # 取步进参数小数位数
        digit = max(start_digit, stop_digit, step_digit)      # 取小数位最大值
        return [(start*10**digit+i*step*10**digit)/10**digit for i in range(int((stop-start)//step))]
    
    
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
    
    print("参数择优完毕")
    #####建模#######
    lr = LR(
            **lr_Bayes.best_params_
        ).fit(Xtrain, Ytrain)#线性回归
    ridge = Ridge(
        **ridge_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#岭回归
    
    adb = ADBR(
        **adb_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#AdaBoost
    
    bpnn = MLPRegressor(
        **bpnn_Grid.best_params_
                        ).fit(Xtrain,Ytrain)#多层感知机
    cbr = CTBR(
        # **cbr_Bayes.best_params_
        **cbr_Grid.best_params_
               ).fit(Xtrain,Ytrain)#CatBoost
    etr = ETR(
        **etr_Bayes.best_params_
              ).fit(Xtrain,Ytrain)#极端随机树
    lgb = LGBR( 
        **lgb_Bayes.best_params_
               ).fit(Xtrain,Ytrain)#LightGBM
    rfr = RFR(
        **rfr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#随机森林
    svr = svm.SVR(
        **svr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#支持向量机
    xgb = XGBR(
        **xgb_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#XGBoost
    lasso = Lasso(
        **lasso_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#Lasso
    dtr = DTR(
        **dtr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#决策树回归
    gbr = GBR(
        **gbr_Bayes.best_params_
        ).fit(Xtrain,Ytrain)#梯度提升
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
    
    
    #预测
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
        #模型准确度
        model_evaluations = []
        model_evaluations.append(model_pass_check_list[i].score(Xvalid,Yvalid))
        model_evaluations.append(mean_squared_error(Yvalid,model_pass_check_list[i].predict(Xvalid))**0.5)
        model_evaluations.append(mean_squared_log_error(abs(Yvalid),abs(model_pass_check_list[i].predict(Xvalid))))
        model_evaluations.append(median_absolute_error(Yvalid,model_pass_check_list[i].predict(Xvalid)))
        model_evaluations.append(mean_absolute_error(Yvalid,model_pass_check_list[i].predict(Xvalid)))
        model_evaluations.append(explained_variance_score(Yvalid,model_pass_check_list[i].predict(Xvalid))) 
        #模型稳定性
        model_evaluations.append(cross_val_score(model_pass_check_list[i],Xtemp, Ytemp,cv=10).mean())
        model_evaluations.append(-(-(cross_val_score(model_pass_check_list[i],Xtemp, Ytemp,cv=10
                                                  ,scoring = "neg_mean_squared_error").mean()))**0.5)
        #----
        Ytemp_pred = cross_val_predict(model_pass_check_list[i], Xtemp, Ytemp, cv=10)                               
        model_evaluations.append(mean_squared_log_error(abs(Ytemp), abs(Ytemp_pred)))
        #----
        model_evaluations.append(cross_val_score(model_pass_check_list[i],Xtemp, Ytemp,cv=10
                                                  ,scoring = "neg_median_absolute_error").mean())
        model_evaluations.append(cross_val_score(model_pass_check_list[i],Xtemp, Ytemp,cv=10
                                                  ,scoring = "neg_mean_absolute_error").mean())
        model_evaluations.append(cross_val_score(model_pass_check_list[i],Xtemp, Ytemp,cv=10
                                                  ,scoring = "explained_variance").mean())
        #导入评价标准矩阵
        acc_data.append(model_evaluations)
    #转化为DataFrame格式
    acc_df = pd.DataFrame(acc_data,index=model_pass_check_list,columns=evaluations,dtype=float)


    #模型排序
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
    #总的排序
    acc_df['sort_score'] = acc_df.iloc[:,12:].sum(axis=1)
    #按评分排序，模型越好越靠前
    acc_df.sort_values(by='sort_score', inplace=True, ascending=True)
    
    return acc_df,models

#%% 计算与输出结果
import pandas as pd
import os
import joblib
import shutil

# 假设你的数据集文件夹名称为datasets
dataset_folder = "datasets"
output_folder = "results"  # 用于保存结果的文件夹名称

# 检查结果文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

name_of_models = ['lr','ridge','adb','bpnn','cbr',
                  'etr','lgb','rfr','svr','xgb',
                  'lasso','dtr','gbr','br']

# 遍历数据集文件夹中的文件
for filename in os.listdir(dataset_folder):
    if filename.endswith(".csv"):
        dataset_name = os.path.splitext(filename)[0]  # 提取文件名（不包含扩展名）作为数据集名称

        dataset_path = os.path.join(dataset_folder, filename)  # 构建数据集文件的完整路径
        output_dataset_folder = os.path.join(output_folder, dataset_name)  # 构建结果文件夹的完整路径

        # 检查结果文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_dataset_folder):
            os.makedirs(output_dataset_folder)

        data = pd.read_csv(dataset_path)  # 读取数据集为 DataFrame

        # 遍历Trainset_Size_list
        Trainset_Size_list=[0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        for trainset_size in Trainset_Size_list:
            trainset_folder = os.path.join(output_dataset_folder, f"Trainset_Size_{trainset_size}")
            
            # 检查Trainset_Size文件夹是否存在，如果不存在则创建
            if not os.path.exists(trainset_folder):
                os.makedirs(trainset_folder)

            # 在此处调用 core() 函数进行计算，将结果保存到结果文件夹中
            result = Core(data, trainset_size)
            acc_df, models = result

            # 将评价指标保存为CSV文件
            acc_csv_filename = os.path.join(trainset_folder, "modelAcc.csv")
            acc_df.to_csv(acc_csv_filename)

            save_directory = os.path.join(trainset_folder, "models")  # 模型保存的文件夹名称
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # 遍历模型列表并保存为pickle文件
            for name, model in zip(name_of_models, models):
                model_name = f'model_{name}.pickle'
                model_path = os.path.join(save_directory, model_name)

                with open(model_path, 'wb') as model_file:
                    pickle.dump(model, model_file)

                print(f"保存模型 {model_name}")

            print(f"Trainset_Size {trainset_size} 的模型保存完成。")

            # 将评价指标CSV文件复制到模型文件夹中
            acc_csv_destination = os.path.join(save_directory, "modelAcc.csv")
            shutil.copyfile(acc_csv_filename, acc_csv_destination)
            print(f"复制评价指标文件到 {acc_csv_destination} 完成")

            # 将测试数据集保存为CSV文件
            data_filename = os.path.join(trainset_folder, filename)
            data.to_csv(data_filename, index=False)
