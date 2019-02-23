

labelList = []
paraList = []
scoreList = []
min_scoreList = []
word_list1 = ['also', 'and','off', 'about','for','what','which','when','where', 'over', 'by', 'it', 'its']
word_list2 = ['you', 'he', 'she', 'we', 'yours', 'ours', 'our', 'your', 'his', 'her','him','they','their', 'theirs']
def scorePara(dataType):
    for para in paraList:
        paraLen = len(para)
        if paraLen <= 1:
            print(para)
        paraScore = []          # the scores of the para
        for i in range(0,paraLen-1):
            curr = para[i]      # the curr sentence
            next = para[i+1]    # the next sentence

            score = 0.0
            for word in curr:
                if word in next:
                    if word in word_list1:
                        score += 0.2
                    elif word in word_list2:
                        score += 0.5
                    else:
                        score += 3
                else:
                    score += 0.1
            score /= (len(curr)+len(next))  #the score for the coherence between curr and next
            paraScore.append(score)
        scoreList.append(paraScore)

    for paraScore in scoreList:
        minScore = min(paraScore)
        min_scoreList.append(minScore)


    #saveScore(dataType)

def saveScore(dataType):
    scoreFileName = 'Data/dataScore/'+dataType + '_score.txt'
    scoreFile = open(scoreFileName, 'w')
    for paraScore in scoreList:
        for score in paraScore:
            scoreFile.write(str(score))
            scoreFile.write(' ')
        scoreFile.write('\n')
    scoreFile.close()

def getData(dataType):
    labelFileName = 'Data/processedData/'+dataType+'_label.txt'
    labelFile = open(labelFileName, 'r')
    line = labelFile.readline()
    line = line.split()
    for label in line:
        labelList.append(label)
    labelFile.close()

    textFileName = 'Data/processedData/'+ dataType+'_relevance.txt'
    textFile = open(textFileName, 'r')

    sentc_end = ['!' '?', '.']
    while True:
        line = textFile.readline()
        if not line:
            break
        para = []
        line = line.split()
        line_len = len(line)
        line_count = 0
        sentc = []
        for word in line:
            if word in sentc_end:
                para.append(sentc)
                sentc = []
            else:
                sentc.append(word)
                if line_count == line_len - 1:
                    para.append(sentc)
                    sentc = []
            line_count += 1

        paraList.append(para)

    textFile.close()

def test():
    print("labelList len: %d"%(len(labelList)))
    print("min_scoreList len: %d"% len(min_scoreList))

    listLen = len(min_scoreList)
    corr = 0
    for i in range(0, listLen):
        if min_scoreList[i] > 0.0814 and int(labelList[i])==1:
            corr += 1
        elif min_scoreList[i] <= 0.0814 and int(labelList[i])==0:
            corr += 1
    print(corr/len(labelList))


if __name__ == "__main__":
    dataType = ["train", "valid", "test"]
    getData('valid')
    scorePara('valid')
    test()
