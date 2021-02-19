import numpy as np
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import datasets
import csv
import random



def get_list(filetpath):
    # get the csv file
    #filetpath = "train_set.csv"
    with open(filetpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = []
        for line in reader:
            data.append(line)
        print(header)
        print(data)

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

        # the length of the dict
        len_dict = len(dict_x)

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
            list_22[i1].append(list(va1))
            #print(list_22[i1])
            # clear
            for w in dict_empt.keys():
                dict_empt[w]=dict_empt.get(w,0)*0

        # label dataset
        label = [i[1] for i in data]
    return list_22, label

# train dataset
test_sen, test_lab = get_list("train_set.csv")


# validation dataset
valid_sen, valid_lab = get_list("validation_set.csv")
print(valid_sen[1])

class KNN():
    ##给出训练数据以及对应的类别
    def createDataSet():
        group = np.array(test_sen)
        labels =np.array(test_lab)

        for g,l in zip(group, labels):
            print(l, ":", g)
        return group,labels

    ###通过KNN进行分类
    def classify(input,dataSet,label,k):


        dist += abs(r1[k] - r2[k])


        ##对距离进行排序
        sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

        classCount={}
        for i in range(k):
            voteLabel = label[sortedDistIndex[i]]
            ###对选取的K个样本所属的类别个数进行统计
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        ###选取出现的类别次数最多的类别
        maxCount = 0
        for key,value in classCount.items():
            if value > maxCount:
                maxCount = value
                classes = key

        return classes


dataSet,labels = KNN.createDataSet()
input = array([1.1,0.3])
K = 3
output = KNN.classify(input,dataSet,labels,K)
print("test item:",input,"\nclass:",output)