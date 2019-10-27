# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

data = pd.read_csv('sonar.all-data.csv', header=None)
data = data.sample(frac=1)
labels = []

for i in data[60]:
    if i == 'M':
        labels.append(1)
    else:
        labels.append(0)
data[60] = pd.Series(labels)

x_train = data.iloc[0:180, :60].as_matrix()
y_train = data.iloc[0:180, 60].as_matrix()
x_test = data.iloc[180:, :60].as_matrix()
y_test = data.iloc[180:, 60].as_matrix()

base_models = [KNeighborsClassifier, Perceptron, DecisionTreeClassifier]
kf = KFold(n_splits=3, shuffle=True, random_state=2017)

x_first=[]
data_test=[]
y_first=[]
for i, (train_index, val_index) in enumerate(kf.split(x_train)):
    prediction = []
    testing=[]
    x_tra, y_tra = x_train[train_index], y_train[train_index]  # training dataset
    x_val, y_val = x_train[val_index], y_train[val_index]  # validatation dataset
    y_first.append(y_val)
    for j, m in enumerate(base_models):
        model = m()
        model = model.fit(x_tra, y_tra)
        prediction.append(model.predict(x_val))
        testing.append(model.predict(x_test))
    x_first.append(np.array(prediction).T)
    data_test.append(np.array(testing).T)
#第二层训练数据
x_second=np.concatenate(x_first,axis=0)
y_second=np.concatenate(y_first,axis=0)

#第二层测试数据
input_second=np.zeros(data_test[0].shape)
#求平均值
for d in data_test:
    input_second+=d
#四舍五入
input_second=np.round(input_second/3)

final_model=LogisticRegression()
final_model=final_model.fit(x_second,y_second)
#这个地方出问题了LogisticRegression需要的输入是一个三维向量，而输入变成了一个60维度的向量因此不对
#因此需要在前面的两个for循环当中对测试数据也转换成28*3的结构否则没有办法作为输入数据。
y_pred=final_model.predict(input_second)
#y_pred.shape=(84,), y_test.shape=(28,4)。 因此存在问题
print(roc_auc_score(y_test,y_pred))


# for m in base_models:
#     kf = KFold(n_splits=3, random_state=2017, shuffle=True)
#     #oof_train = np.zeros((x_train.shape[0],))
#     #oof_test=np.zeros((x_test.shape[0],))
#     #oof_test_all_fold=np.zeros((x_test.shape[0],3))
#     x_first_train=[]
#     y_first_train=[]
#     x_first_test=[]
#     y_first_test=[]
#     for i, (train_index, val_index) in enumerate(kf.split(x_train)):
#         print('{0} fold, train {1}, val {2}'.format(
#             i,
#             len(train_index),
#             len(val_index)
#         ))
#         x_tra, y_tra = x_train[train_index], y_train[train_index]
#         x_val, y_val = x_train[val_index], y_train[val_index]
#         model = m()
#         model = model.fit(x_tra, y_tra)
#         x_first_train.append(model.predict(x_val))
#         y_first_train.append(y_val)
#         x_first_test.append(model.predict(x_test))
#         #oof_train[val_index]=model.predict(x_val)
#         #oof_test_all_fold[:,i]=model.predict(x_test)
