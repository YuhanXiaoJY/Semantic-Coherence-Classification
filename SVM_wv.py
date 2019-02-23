from sklearn import svm
import numpy as np
import time


if __name__ == '__main__':
    trainText = np.load('Data/word2vector/train_sentc_word2vec.npy')
    train_labelFileName = 'Data/processedData/train_label.txt'
    train_labelFile = open(train_labelFileName, 'r')
    line = train_labelFile.readline()
    train_labelList = line.split()
    train_labelList = [int(label) for label in train_labelList]
    train_labelFile.close()

    time0 = time.time()

    clf = svm.NuSVC(max_iter=1000, nu=0.4, cache_size=1500, gamma='auto')
    clf.fit(trainText, train_labelList)
    time1 = time.time()
    print("training done: %fs" % (time1-time0))

    validText = np.load('Data/word2vector/valid_sentc_word2vec.npy')
    valid_labelFileName = 'Data/processedData/valid_label.txt'
    valid_labelFile = open(valid_labelFileName, 'r')
    line = valid_labelFile.readline()
    valid_labelList = line.split()
    valid_labelList = [int(label) for label in valid_labelList]
    valid_labelFile.close()

    print(clf.score(validText, valid_labelList))
