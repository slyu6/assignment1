import numpy as np
import matplotlib.pyplot as plt
import operator

def creatDataSet():
    group = np.array([[1.0,2.0],
                      [1.2,0.1],
                      [0.1,1.4],
                      [0.3,3.5],
                      [1.1,1.0],
                      [0.5,1.5]])
    lebals = np.array(['A','A','B','B','A','B'])
    print(group)
    return group,lebals

def KNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis == 'E' or dis =='M','dis must E or M,E为欧拉距离，M为曼哈顿距离'
    num_test = Y_test.shape[0]
    leballist = []
    if dis == 'E':
        for i in range(num_test):
            distances = np.sqrt(np.sum(((X_train - np.tile(Y_test[i],(X_train.shape[0],1))) ** 2 ),axis = 1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i],0) + 1
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            leballist.append (sortedClassCount[0][0])
        return np.array(leballist)
    else:
        for i in range(num_test):
            distances = np.sum(np.abs(X_train[-1,1] - Y_test[-1,1]) + np.abs(X_train[-1,0] - Y_test[-1,1]))
            print(distances)
            nearest_k = np.argsort(distances)
            print(nearest_k)
            topK = nearest_k[:k]
            print(topK)
            classCount = {}
            for i in topK:
                print(i)
                classCount[x_train[i]] = classCount.get(x_train[i],0) + 1
                print(classCount)
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            print(sortedClassCount)
            leballist.append (sortedClassCount[0][0])
        return np.array(leballist)

if __name__ == "__main__":
    group,lebals = creatDataSet()
    y_test_res = KNN_classify(1,'M',group,lebals,np.array([[1.0,2.1],[0.4,2.0]]))
    print(y_test_res)
    plt.show()