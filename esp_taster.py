import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Constants
epoch_num = 5000

# Data retrieval
train = np.genfromtxt('winedata/train.csv', delimiter=',')
train = train[1:]  # Waste the header line
train_data, train_tags = train[:, :-1], train[:, -1]
train_tags_one_hot = np_utils.to_categorical(train_tags)  # int to one-hot

# Set the attributes of a model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=11))  # hidden layer 1
model.add(Dense(units=24, activation='relu', input_dim=32))  # hidden layer 2
model.add(Dense(units=16, activation='relu', input_dim=24))  # hidden layer 3
model.add(Dense(units=9, activation='softmax'))  # output layer
model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_tags_one_hot, epochs=epoch_num)

# Show the accuracy for the 'train' data set
predictions = model.predict_classes(train_data)
acc = 0
for i in range(0, len(train_tags)):
    if train_tags[i] == predictions[i]:
        acc += 1
print("Number of epochs: " + str(epoch_num))
print("Number of correct predictions: " + str(acc) + "/" + str(i+1))
print("Accuracy: " + str(acc / (i+1)))
