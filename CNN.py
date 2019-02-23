import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Conv1D, MaxPooling1D

def build_model(wvMatrix):
    embedding = Embedding(input_dim=54841, output_dim=100, weights=[wvMatrix], input_length=800, trainable=False)
    seqInput = Input(shape=(800,), dtype='int32')
    sequences = embedding(seqInput)
    x = Conv1D(128, 5, activation='relu')(sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(27)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    last = Dense(1, activation='sigmoid')(x)

    model = Model(seqInput, last)
    return model


if __name__ == "__main__":
    trainText = np.load('Data/word2vector/train_index.npy')
    train_labelFileName = 'Data/processedData/train_label.txt'
    train_labelFile = open(train_labelFileName, 'r')
    line = train_labelFile.readline()
    train_labelList = line.split()
    train_labelList = [int(label) for label in train_labelList]
    train_labelFile.close()

    validText = np.load('Data/word2vector/valid_index.npy')
    valid_labelFileName = 'Data/processedData/valid_label.txt'
    valid_labelFile = open(valid_labelFileName, 'r')
    line = valid_labelFile.readline()
    valid_labelList = line.split()
    valid_labelList = [int(label) for label in valid_labelList]
    valid_labelFile.close()

    wvMatrix = np.load('Data/word2vector/wv_for_weight_matrix.npy')

    model = build_model(wvMatrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainText, train_labelList, validation_data=(validText, valid_labelList),
              epochs=6, batch_size=64)

    testText = np.load('Data/word2vector/test_index.npy')
    array = model.predict(testText)


    file = open('Data/result/result.txt','w')
    count =0
    for label in array:
        if label[0] > 0.5:
            file.write('1'+'\n')
            count+=1
        else:
            file.write('0'+'\n')

    file.close()
    print(count)


