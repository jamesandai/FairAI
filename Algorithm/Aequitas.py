import sys
sys.path.append("../")

import numpy as np
import tensorflow as tf
import copy
from Basic_Class.Load_Data import Load_Data as LD
from Basic_Class.Utils.Config import census, credit, bank

class Aequitas:
    def __init__(self,dataset, sensitive_param, model_path, max_global, max_local, step_size):
        self.data = {"census": LD.Load_Census(), "credit": LD.Load_Credit(), "bank": LD.Load_Bank()}
        self.data_config = {"census": census, "credit": credit, "bank": bank}
        params = self.data_config[dataset].params

    def check_for_error_condition(self,conf, sess, x, preds, t, sens):
        """
        Check whether the test case is an individual discriminatory instance
        :param conf: the configuration of dataset
        :param sess: TF session
        :param x: input placeholder
        :param preds: the Algorithm's symbolic output
        :param t: test case
        :param sens: the index of sensitive feature
        :return: the value of sensitive feature
        """
        t = np.array(t).astype("int")
        label = model_argmax(sess, x, preds, np.array([t]))

        # check for all the possible values of sensitive feature
        for val in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1):
            if val != int(t[sens - 1]):
                tnew = copy.deepcopy(t)
                tnew[sens - 1] = val
                label_new = model_argmax(sess, x, preds, np.array([tnew]))
                if label_new != label:
                    return val
        return t[sens - 1]

    def aequitas(self,dataset, sensitive_param, model_path, max_global, max_local, step_size):
        """
           The implementation of AEQUITAS_Fully_Connected
           :param dataset: the name of testing dataset
           :param sensitive_param: the name of testing dataset
           :param model_path: the path of testing Algorithm
           :param max_global: the maximum number of samples for global search
           :param max_local: the maximum number of samples for local search
           :param step_size: the step size of perturbation
           :return:
           """
        data = {"census": census_data, "credit": credit_data, "bank": bank_data}
        data_config = {"census": census, "credit": credit, "bank": bank}
        params = data_config[dataset].params

        # hyper-parameters for initial probabilities of directions
        init_prob = 0.5
        direction_probability = [init_prob] * params
        direction_probability_change_size = 0.001

        # hyper-parameters for features
        param_probability = [1.0 / params] * params
        param_probability_change_size = 0.001

        # prepare the testing data and Algorithm
        X, Y, input_shape, nb_classes = data[dataset]()
        model = dnn(input_shape, nb_classes)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        preds = model(x)
        tf.set_random_seed(1234)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # store the result of fairness testing
        global_disc_inputs = set()
        global_disc_inputs_list = []
        local_disc_inputs = set()
        local_disc_inputs_list = []
        tot_inputs = set()

        # initial input
        if dataset == "census":
            initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
        elif dataset == "credit":
            initial_input = [2, 24, 2, 2, 37, 0, 1, 2, 1, 0, 4, 2, 2, 2, 1, 1, 2, 1, 0, 0]
        elif dataset == "bank":
            initial_input = [3, 11, 2, 0, 0, 5, 1, 0, 0, 5, 4, 40, 1, 1, 0, 0]
        minimizer = {"method": "L-BFGS-B"}








