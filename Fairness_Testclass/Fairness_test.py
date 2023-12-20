import sys
import os
from tensorflow.python.platform import flags,app
import tensorflow as tf

import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Basic_Class.Load_Data import Load_Data
from Basic_Class.Model.tutorial_models import dnn
from Basic_Class.Utils.Utils_tf import model_prediction, model_argmax, model_loss
from Fairness_Evalate import Fairness_Evalute
class Fairness_test:
    def __init__(self) -> None:
        pass

    def DNN_test(self, X, Y, input_shape, nb_classes, dsname, model_path):
        tf.set_random_seed(1234)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        model = dnn(input_shape, nb_classes)
        preds = model(x)
        saver = tf.train.Saver()
        model_path = model_path + dsname + "/999/test.model"#模型保存地址，999是训练次数，所以会有这个文件夹，可以根据定义的训练次数来改
        saver.restore(sess, model_path)
        data_size = len(X)
        Y_size = y.shape[1]
        y_predict = np.zeros((data_size,Y_size))
        for i in range(data_size):
            input_X = X[i].reshape((1,13))
            probs = model_prediction(sess, x, preds, input_X)[0]
            y_predict[i] = probs.tolist()
        for i in range(data_size):
            max_index = np.argmax(y_predict[i],axis=0)
            for j in range(Y_size):
                if j != max_index:
                    y_predict[i][j] = 0
                else:
                    y_predict[i][j] = 1
        return y_predict

if __name__ == "__main__":
    Ft = Fairness_test()
    LD = Load_Data()
    FLAGS = flags.FLAGS
    path = "./Datasets/Numerical_Data/census"
    X, Y, input_shape, nb_classes = LD.Load_Census(path)
    flags.DEFINE_string('dsname', 'census', 'the name of dataset')
    flags.DEFINE_string('model_path', './Generate_Data/models/', 'the path for testing model')

    y_predict = Ft.DNN_test(X, Y, input_shape, nb_classes,FLAGS.dsname,FLAGS.model_path)


    FE = Fairness_Evalute()
    sensitive_param = 8
    res = FE.independence(X,sensitive_param,Y)
    se = FE.separation(X,sensitive_param, Y, y_predict)
    su = FE.sufficiency(X,sensitive_param, Y, y_predict)
    print(res,se,su)