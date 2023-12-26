import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(sys.path)
from sklearn.cluster import KMeans
# from sklearn.externals import joblib
import joblib
import tensorflow as tf
from tensorflow.python.platform import flags
from Basic_Class.Load_Data import Load_Data
class Cluster:
    def cluster(dataset, cluster_num=4):
        """
        Construct the K-means clustering Algorithm to increase the complexity of discrimination
        :param dataset: the name of dataset
        :param cluster_num: the number of clusters to form as well as the number of
                centroids to generate
        :return: the K_means clustering Algorithm
        """
        LD = Load_Data()
        """
        构建 K-means 聚类模型以增加判别的复杂性
        参数 dataset：数据集名称
        参数 cluster_num：要形成的聚类数量以及中心点的数量
        返回：K_means 聚类模型
        """
        path = ["./Datasets/Numerical_Data/census", "./Datasets/Numerical_Data/credit",
                "./Datasets/Numerical_Data/bank"]
        datasets_dict = {"census": LD.Load_Census(path[0]), "credit": LD.Load_Credit(path[1]), "bank": LD.Load_Bank(path[2])}
        if os.path.exists('./Generate_Data/Clusters/' + dataset + '.pkl'):
            clf = joblib.load('./Generate_Data/Clusters/' + dataset + '.pkl')
        else:
            X, Y, input_shape, nb_classes = datasets_dict[dataset]
            clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
            joblib.dump(clf, './Generate_Data/Clusters/' + dataset + '.pkl')
        return clf