Algorithm:存各类的算法

Basic_Class:存算法运行需要的必备类，包括Lime线性求解器，SG算法需要的，Model是DNN模型训练，Utils包括Config是每个数据集样本的配置（值域、类别名等）、Utils步长信息、Utils_tf是关于tf中的损失函数、模型训练等
Cluster是聚类算法，Data_Preprocess将文本数据集转换为数值数据集，Load_Data读取数据集类

Datasets：原始数据集，包括未转换为数值的数据集

Evaluate_Metric：评估指标

Generate_Data：模型训练生成的参数、聚类结果、生成的样本集合(numpy格式)

