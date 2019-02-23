from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
import re

def read_stopwords(filePath):
    file = open(filePath, 'r')
    stopwordsList = []
    while True:
        line = file.readline()
        if not line:
            break
        line = line.split()
        stopwordsList.append(line[0])

    file.close()
    return stopwordsList

def preprocess(dataType):
    time0 = time.time()
    stopwordsPath = 'Data/stopwordsList/stopwords.txt'
    stopwordsList = read_stopwords(stopwordsPath)

    raw_fileName = 'Data/rawData/'+dataType +'_data'
    raw_file = open(raw_fileName,'r')

    output_fileName = 'Data/processedData/' + dataType + '_relevance.txt'
    output_file = open(output_fileName,'w')

    labelList = []
    textPattern = re.compile('(.*)"text": "(.*)"(.*)')
    numberList = ['0', '2', '1', '3', '4', '5', '6', '7', '8', '9']
    lastFilter = ['a', '%', '#', ':']    #the last filter

    while True:
        line = raw_file.readline()
        if not line:
            break
        if dataType != 'test':
            labelList.append(line[11])

        textMatch = re.match(textPattern, line)
        text = textMatch[2]
        wordList = text.split(r' ')


        lemmatizer = WordNetLemmatizer()    # lemmatize the word
        porter_stemmer = PorterStemmer()    # extract the stem
        wordList = [porter_stemmer.stem(lemmatizer.lemmatize(word)).lower() for word in wordList if word not in stopwordsList
                    and word[0] not in numberList]
        # remove the number, remove the stop words, #get the stem, lemmatize the word, and lower-case the word

        for word in wordList:
            if word not in lastFilter:
                output_file.write(word + ' ')
        output_file.write('\n')

    raw_file.close()
    print("%s_relevance.txt done!"%dataType)
    time1 = (time.time() - time0)
    print("%s_relevance.txt consumes %fs "%(dataType,time1))

    if dataType != 'test':
        labelFileName = 'Data/processedData/' + dataType + '_label.txt'
        labelFile = open(labelFileName, 'w')
        for label in labelList:
            labelFile.write(label + ' ')
        labelFile.close()
        print("%s_label.txt done!"%dataType)
        time2 = (time.time() - time1 - time0)
        print("%s_label.txt consumes %fs " % (dataType, time2))

if __name__ == "__main__":
    dataType_List = ['valid', 'train', 'test']
    preprocess('valid')
    # labelFileName = 'Data/processedData/' + 'valid' + '_label.txt'
    # labelFile = open(labelFileName, 'r')
    # line = labelFile.readline()
    # labelList = line.split()
    # print(labelList)