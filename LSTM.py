import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, Flatten, Bidirectional

def build_model(wvMatrix):
    
    model = Sequential()
    model.add(Embedding(input_dim=54841, output_dim=100, weights=[wvMatrix], input_length=800, mask_zero=False, trainable=False))
    model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

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
    model.fit(trainText, train_labelList, validation_data=(validText, valid_labelList), epochs=5,
              batch_size = 64)