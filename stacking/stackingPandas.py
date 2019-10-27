import pandas as pd
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron


def cross_validation_split(data, n_folds):
    fold_size = int(data.shape[0] / n_folds)
    data_split=dict()
    for i in range(n_folds):
        fold=pd.DataFrame()
        while fold.shape[0]<fold_size:
            index=randrange(data.shape[0])
            fold=fold.append(data.loc[index],ignore_index=True)
        data_split[i]=fold
    return data_split

def KNN(training,labels,test,n_neighbors,):
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(training,labels)
    prediction=knn.predict(test)
    return prediction

def PPN(training,labels,test):
    ppn=Perceptron(
        n_iter=40,
        eta0=0.1,
        random_state=0
    )
    ppn.fit(training,labels)
    prediction=ppn.predict(test)
    return prediction

n_folds = 3
data = pd.read_csv('sonar.all-data.csv', header=None)
data_split=cross_validation_split(data,n_folds)
knn_training=data_split[0].iloc[:,:60]
knn_label=data_split[0].iloc[:,60]
ppn_training=data_split[1].iloc[:,:60]
ppn_label=data_split[1].iloc[:,60]
test_data=data_split[2].iloc[:,:60]
test_label=data_split[2].iloc[:,:60]

knn_prediction=KNN(knn_training,knn_label,test_data,3)
ppn_prediction=PPN(ppn_training,ppn_label,test_data)


