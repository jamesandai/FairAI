import os

import pandas as pd

from Evaluate_Index.datasets import StandardDataset

# 继承自 StandarDataset
class BankDataset(StandardDataset):
    """Bank marketing Dataset.
    """

    def __init__(self, label_name='y', favorable_classes=['yes'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age >= 25` and unprivileged is `age < 25` as in
        :obj:`GermanDataset`.
        """
        # 定义数据集文件所在的位置 .././Datasets/Raw_data/Bank_Raw/bank-additional-full.csv
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..','.' ,'Datasets', 'Raw_data', 'Bank_Raw', 'bank-additional-full.csv')

        # 读取csv格式的文件，分隔符为;，使用自定义缺失值
        try:
            df = pd.read_csv(filepath, sep=';', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'bank'))))
            import sys
            sys.exit(1)

        # 调用父类 即 StandardDataset 类的构造函数，继承父类的属性和方法，并传入参数初始化父类的属性
        super(BankDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
