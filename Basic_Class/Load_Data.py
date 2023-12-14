import numpy as np
import sys
import os
#读取数据集类，X, Y, input shape and number of classes（属性，标签，维度，类别）
class Load_Data:
    #读取Census
    def Load_Census(self,path):
        X = []
        Y = []
        i = 0
        with open(path, "r") as ins:
            for line in ins:
                line = line.strip()
                line1 = line.split(',')
                if (i == 0):
                    i += 1
                    continue
                # L = map(int, line1[:-1])
                L = [int(i) for i in line1[:-1]]
                X.append(L)
                if int(line1[-1]) == 0:
                    Y.append([1, 0])
                else:
                    Y.append([0, 1])
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        input_shape = (None, 13)
        nb_classes = 2

        return X, Y, input_shape, nb_classes

    def Load_Bank(self,path):
        #读取bank数据集
        X = []
        Y = []
        i = 0
        with open(path, "r") as ins:
            for line in ins:
                line = line.strip()
                line1 = line.split(',')
                if (i == 0):
                    i += 1
                    continue
                # L = map(int, line1[:-1])
                L = [int(i) for i in line1[:-1]]
                X.append(L)
                if int(line1[-1]) == 0:
                    Y.append([1, 0])
                else:
                    Y.append([0, 1])
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        input_shape = (None, 16)
        nb_classes = 2

        return X, Y, input_shape, nb_classes

    def Load_Credit(self,path):
        #读取Credit数据集
        """
        Prepare the data of dataset German Credit
        :return: X, Y, input shape and number of classes
        """
        X = []
        Y = []
        i = 0

        with open(path, "r") as ins:
            for line in ins:
                line = line.strip()
                line1 = line.split(',')
                if (i == 0):
                    i += 1
                    continue
                # L = map(int, line1[:-1])
                L = [int(i) for i in line1[:-1]]
                X.append(L)
                if int(line1[-1]) == 0:
                    Y.append([1, 0])
                else:
                    Y.append([0, 1])
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        input_shape = (None, 20)
        nb_classes = 2

        return X, Y, input_shape, nb_classes
