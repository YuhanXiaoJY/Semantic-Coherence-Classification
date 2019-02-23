import numpy as np
from gensim.models import word2vec
from gensim.models.word2vec import Text8Corpus
import time

def testwv(word):
    modelFile = 'Data/word2vector/raw_word2vec.model'
    model = word2vec.Word2Vec.load(modelFile)
    print(model.most_similar(word))

def SGM():
    time0 = time.time()
    textFile = 'Data/processedData/train_processed.txt'
    text = word2vec.Text8Corpus(textFile)

    model = word2vec.Word2Vec(sentences=text,window = 8, min_count = 3, sg = 1)
    model.save('Data/word2vector/raw_word2vec.model')

    time1 = time.time() - time0
    print("raw word2vec trained: %fs" % time1)

def fcount(line):      # compute the frequency of the word in the paragraph
    fDict = {}
    for word in line:
        fDict[word] = [0, 0.0]
    for word in line:
        fDict[word][0] += 1     # the count of the word in this line
    Len = len(line)
    for word in line:
        fDict[word][1] = fDict[word][0]/Len # the frequency of the word
    return fDict

def savewv(vectorList, dataType):
    time0 = time.time()
    fileName = 'Data/word2vector/'+dataType +'_sentc_word2vec.npy'
    vectorList = np.array(vectorList)
    np.save(fileName,vectorList)
    time1 = time.time()
    print("%s saved: consumes %fs" % (dataType + '_sentc_word2vec.npy', time1 - time0))


def wordVecHandler(dataType):
    time0 = time.time()
    modelFile = 'Data/word2vector/raw_word2vec.model'
    model = word2vec.Word2Vec.load(modelFile)
    wvDict = model.wv

    wordVector = []
    wordVector.append(np.zeros(100))
    for word in wvDict.vocab.keys():
        wordVector.append(wvDict[word])
    print(len(wordVector))
    wordVector = np.array(wordVector)
    np.save('Data/word2vector/wv_for_weight_matrix.npy', wordVector)

    vectorList = []
    time1 = time.time()
    print("raw_word2vec loaded: consumes %fs" % (time1-time0))

    fileName = 'Data/processedData/' + dataType + '_processed.txt'
    file = open(fileName, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        line = line.split()
        countDict = fcount(line)
        sentence_embdd = np.zeros(100)
        for word in countDict.keys():
            if word in wvDict:
                sentence_embdd += countDict[word][1] * wvDict[word] # weighted add

        vectorList.append(sentence_embdd)

    time2= time.time()
    print("sentence embedding done: consumes %fs" % (time2 - time1))
    savewv(vectorList, dataType)

    CL_wvHandler(dataType)

def CL_wvHandler(dataType):
    modelFile = 'Data/word2vector/raw_word2vec.model'
    model = word2vec.Word2Vec.load(modelFile)

    wvList = []
    wvList.append('')
    for word in model.wv.vocab.keys():
        wvList.append(word)


    filename = 'Data/processedData/' + dataType + '_processed.txt'
    file = open(filename, 'r')
    myIndex = []

    while True:
        line = file.readline()
        tmp = np.zeros(800)
        if not line:
            break

        count = 0
        text = line.split()
        for word in text:
            if word in model.wv.vocab.keys():
                tmp[count] = wvList.index(word)
                count += 1
        myIndex.append(tmp)

    myIndex = np.array(myIndex)
    # print(myIndex.shape)
    indexFilename = 'Data/word2vector/' + dataType + '_index.npy'
    np.save(indexFilename, myIndex)

if __name__ == '__main__':
    SGM()
    #wordVecHandler("train")
    CL_wvHandler("test")
    # testwv('damn')
