import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from Basic_Class.Load_Data import Load_Data

import numpy as np
import pandas as pd

X, Y, input_shape, nb_classes = Load_Data.Load_Census()
Y = Y[:,0]

class fairness_eva:

    def __init__(self):
        self.LD = Load_Data()
        self.path = ["./Datasets/Numerical_Data/census", "./Datasets/Numerical_Data/credit",
                "./Datasets/Numerical_Data/bank"]
        pass

    # indenpendence
    def independence(self,x_tranin, sensitive_param, y_target):
        sens_var = list(x_tranin[:,sensitive_param])
        target_var = list(y_target)
        sens_lvls = np.unique(sens_var)
        target_lvls = np.unique(target_var)


        data = pd.DataFrame({"sens":sens_var, "target":target_var})


        total_count = data.shape[0]
        target1_count = len(data[data['target'] == target_lvls[0]])



        P_1 = ((data.loc[(data['sens'] == sens_lvls[0]) & (data['target'] == target_lvls[0])].shape[0]) / target1_count) * (target1_count / total_count) / ((len(data[data['sens'] == sens_lvls[0]])) / total_count)
        P_2 = ((data.loc[(data['sens'] == sens_lvls[1]) & (data['target'] == target_lvls[0])].shape[0]) / target1_count) * (target1_count / total_count) / ((len(data[data['sens'] == sens_lvls[1]])) / total_count)


        res = abs(P_1 - P_2)


        return res


    def separation(self,x_tranin,sensitive_param, y_target, y_predict):
        sens_var = list(x_tranin[:,sensitive_param])
        target_var = list(y_target)
        predict_var = list(y_predict)
        sens_lvls = np.unique(sens_var)
        target_lvls = np.unique(target_var)

        data = pd.DataFrame({"sens":sens_var, "target":target_var, "predicted": predict_var})

        #非保护属性
        data_un = data[data['sens'] == sens_lvls[0]]
        FN_un = data.loc[(data_un['target'] == target_lvls[0]) & (data_un['predicted'] == target_lvls[1])].shape[0]
        FP_un = data.loc[(data_un['target'] == target_lvls[1]) & (data_un['predicted'] == target_lvls[0])].shape[0]
        TP_un = data.loc[(data_un['target'] == target_lvls[0]) & (data_un['predicted'] == target_lvls[0])].shape[0]
        TN_un = data.loc[(data_un['target'] == target_lvls[1]) & (data_un['predicted'] == target_lvls[1])].shape[0]
        FPR_un = FP_un / (TN_un + FP_un)
        TPR_un = TP_un / (TP_un + FN_un)

        #保护属性
        data_priv = data[data['sens'] == sens_lvls[1]]
        FN_priv = data.loc[(data_priv['target'] == target_lvls[0]) & (data_priv['predicted'] == target_lvls[1])].shape[0]
        FP_priv = data.loc[(data_priv['target'] == target_lvls[1]) & (data_priv['predicted'] == target_lvls[0])].shape[0]
        TP_priv = data.loc[(data_priv['target'] == target_lvls[0]) & (data_priv['predicted'] == target_lvls[0])].shape[0]
        TN_priv = data.loc[(data_priv['target'] == target_lvls[1]) & (data_priv['predicted'] == target_lvls[1])].shape[0]
        FPR_priv = FP_priv / (TN_priv + FP_priv)
        TPR_priv = TP_priv / (TP_priv + FN_priv)

        return abs(((FPR_un - FPR_priv) + (TPR_un - TPR_priv)) /2)


    def sufficiency(self,x_tranin,sensitive_param, y_target, y_predict):
        sens_var = list(x_tranin[:,sensitive_param])
        target_var = list(y_target)
        predict_var = list(y_predict)
        sens_lvls = np.unique(sens_var)
        target_lvls = np.unique(target_var)

        data = pd.DataFrame({"sens":sens_var, "target":target_var, "predicted": predict_var})

        #非保护属性
        data_un = data[data['sens'] == sens_lvls[0]]
        pp_un = (data_un.loc[(data_un['target'] == target_lvls[1]) & (data_un['predicted'] == target_lvls[1])].shape[0]) / (len(data_un[data_un['predicted'] == target_lvls[1]]))

        #受保护属性
        data_priv = data[data['sens'] == sens_lvls[1]]
        pp_priv = (data_priv.loc[(data_priv['target'] == target_lvls[1]) & (data_priv['predicted'] == target_lvls[1])].shape[0]) / (len(data_priv[data_priv['predicted'] == target_lvls[1]]))

        res = abs(pp_un - pp_priv)

        return res