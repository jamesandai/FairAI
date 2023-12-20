
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))



import numpy as np
import pandas as pd
from Basic_Class.Load_Data import Load_Data
from sklearn import model_selection
# import xgboost as xgb

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

from Evaluate_Index.fairness_evalate import fairness_eva

from sklearn import metrics

model = XGBClassifier()
load = Load_Data()

if __name__ == '__main__':
    LD = Load_Data()
    X, Y, input_shape, nb_classes = LD.Load_Census("./Datasets/Numerical_Data/census")
    Y = Y[:,0]
    x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size = 0.3, random_state = 1234)


    # 调参
    from sklearn.model_selection import GridSearchCV

    learning_rate = [0.1, 0.3,]
    subsample = [0.8]
    colsample_bytree = [0.6, 0.8]
    max_depth = [3,5]

    parameters = { 'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree':colsample_bytree,
                'max_depth': max_depth}

    clf = GridSearchCV(model, parameters, n_jobs = -1, verbose = 2, cv=3, scoring='accuracy')
    clf = clf.fit(x_train, y_train)

    # 随机网格后最好的参数
    print(clf.best_params_)

    # 使用重抽样后的数据，对其建模
    clf = XGBClassifier(colsample_bytree = 0.6, learning_rate = 0.3, max_depth= 8, subsample = 0.8)
    clf.fit(x_train,y_train)
    # # 将模型运用到测试数据集中
    y_testpred = clf.predict(np.array(x_test)) 
    #传入的是array
    y_trainpred = clf.predict(np.array(x_train))

    # 返回模型的预测效果
    print('模型的准确率为：\n',metrics.accuracy_score(y_test, y_testpred))
    print('模型的评估报告：\n',metrics.classification_report(y_test, y_testpred))



    independence_res = fairness_eva.independence(x_train, 8, y_train)
    separation_res = fairness_eva.separation(x_test, 8, y_test, y_testpred)
    sufficiency_res = fairness_eva.sufficiency(x_test, 8, y_test, y_testpred)

    print('模型独立性指标为：\n', independence_res)
    print('模型分离性指标为：\n', separation_res)
    print('模型充分性指标为：\n', sufficiency_res)
