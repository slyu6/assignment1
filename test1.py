import csv
import random
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import datasets
from collections import Counter



def get_list(filetpath):
    # get the csv file
    #filetpath = "train_set.csv"
    with open(filetpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = []
        for line in reader:
            data.append(line)
        # print(header)
        # print(data)

        # extract the words and vector
        # record the length of the dataset = 598
        datasum = len(data)


        # combine the 2-dimension to 1-dimension
        a = [i[0] for i in data]

        comlist = []

        # split the words of first content in list
        # combine the words in one list
        for i1 in a:
            alist = i1.split(" ")
            comlist.extend(alist)

        # create a list to store the vec of each word
        # dictionary
        dict_x = {}
        for key in comlist:
            dict_x[key] = dict_x.get(key, 0) + 1

        # create a empty dict
        dict_empt = {}
        dict_empt.update(dict_x)
        for w in dict_empt.keys():
            dict_empt[w]=dict_empt.get(w,0)*0

        # build 2-dim list
        # input training data
        # list_22 training dataset
        list_22 = []
        for i1 in range(datasum):
            # first content in list
            con_first = data[i1][0]
            a = con_first.split(" ")
            #print(a)
            list_22.append([])
            for w in list(a):
                #print(w)
                dict_empt[w]=dict_empt.get(w,0)+1
            va1 = dict_empt.values()
            list_22[i1].extend(list(va1))
            print(list_22[i1])
            # clear
            for w in dict_empt.keys():
                dict_empt[w]=dict_empt.get(w,0)*0

        # label dataset
        label = [i[1] for i in data]
    return list_22, label

# one hot matrix has been produced
# train dataset
train_sen, train_lab = get_list("train_set.csv")

# validation dataset
valid_sen, valid_lab = get_list("validation_set.csv")

# training dataset split
# label split
random.shuffle(train_lab)
n = len(train_lab) // 2
# separate
train_lab_1 = np.array(train_lab[0:n])
train_lab_2 = np.array(train_lab[n:])


# sentence split
random.shuffle(train_sen)
n = len(train_sen) // 2
# separate
train_sen_1 = np.array(train_sen[0:n])
train_sen_2 = np.array(train_sen[n:])


dist_sum = []
# test
print(np.array(train_sen_2[0][0]))
print(np.array(train_sen_2).shape)
print(np.array(train_sen_1).shape)
print(len(train_sen))

# distance *useless*
def dist(r1, r2):
    return np.abs(np.tile(r1, (r2.shape[0], 1)) - r2).sum(axis=1)

#print(dist(np.array(trainset), np.array(testset)))


print("x", train_sen_2[-1, 1])
print("y", train_sen_1[-1, 1])

def KNN_classify(k,X_train,x_train,Y_test):
    num_test = Y_test.shape[0]
    leballist = []
    for i in range(num_test):
        print("x", X_train[-1,1])
        print("y", Y_test[-1,1])

        distances = np.sum(np.abs(X_train[-1,0] - Y_test[-1,1]) + np.abs(X_train[-1,0] - Y_test[-1,0]))
        #distances = np.sum(np.abs(X_train[-1,1] - Y_test[-1,1]) + np.abs(X_train[-1,0] - Y_test[-1,1]))
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
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        print(sortedClassCount)
        leballist.append(sortedClassCount[0][0])
    return np.array(leballist)

x = KNN_classify(3, train_sen_2, train_lab_2, train_sen_1, )
train_sen
print(x)