
import csv

# get the csv file
filetpath = "validation_set.csv"

with open(filetpath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    data = []
    for line in reader:
        data.append(line)
    print(header)
    print(data)

    # extract the words and vector
    # record the length of the dataset
    datasum = len(data)
    print(datasum)

    # combine the 2-dimension to 1-dimension
    a = [i[0] for i in data]
    print(a)
    print("aaa", len(a))

    # chongfu summary
    list_2 = []
    for i in range(len(a)):
        list_1 = a[i].split(" ")
        list_2.extend(list_1)
    print("222", list_2)
    print(len(list_2))

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
    print(dict_x)

    print(len(dict_x))
    # the length of the dict
    len_dict = len(dict_x)

    # create a empty dict
    dict_empt = {}
    dict_empt.update(dict_x)
    for w in dict_empt.keys():
        dict_empt[w]=dict_empt.get(w,0)*0

    # build 2-dim list
    # input training data
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

        #list_22[i1].append(list(va1))
        list_22[i1].extend(list(va1))

        #print(list_22[i1])
        # clear
        for w in dict_empt.keys():
            dict_empt[w]=dict_empt.get(w,0)*0

    # label dataset
    label = [i[1] for i in data]

print(list_22)


    # # 2 dim list
    # list_2 = []
    # for i in range(len_dict):
    #     list_2.append(0)
    # print(list_2)
    # print(len(list_2))

    #dict_x.subtract