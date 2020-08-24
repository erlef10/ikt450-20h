import math
import numpy as np
import random

np.random.seed(12)
random.seed(12)

# import dataset
dataset = np.loadtxt("ecoli.data", delimiter="\t", dtype=str)

# split the dataset
dataset = [row.split() for row in dataset]

# Set 'im' as 1 and 'cp' as 0
dataset = [row[:-1]+[0 if row[-1] == 'cp' else 1] for row in dataset]

# remove the first column as they are all unique anyhow
dataset = [row[1::] for row in dataset]

# convert the strings to floats
dataset = [[float(x) for x in row] for row in dataset]

# do the shuffle
np.random.shuffle(dataset)

# split the dataset into training and validation datasets
splitratio = 0.8

training_dataset = dataset[:int(len(dataset)*splitratio)]
validation_dataset = dataset[int(len(dataset)*splitratio):]

def sigmoid(z):
    if(z<-100):
        return 0
    if(z>100):
        return 1
    return 1.0/math.exp(-z)

weights = [random.uniform(-1, 1) for i in range(14)]

def firstLayer(row,weights):
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0] # mcg
    activation_1 += weights[2]*row[1] # gvh

    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2] # lip
    activation_2 += weights[5]*row[3] # chg

    activation_3 = weights[6]*1
    activation_3 = weights[7]*row[4] # aac
    activation_3 = weights[8]*row[5] # alm1
    activation_3 = weights[9]*row[6] # alm2

    return sigmoid(activation_1), sigmoid(activation_2), sigmoid(activation_3)

def secondLayer(row, weights):
    activation_4 = weights[10]
    activation_4 += weights[11]*row[0] # activation 1
    activation_4 += weights[12]*row[1] # activation 2
    activation_4 += weights[13]*row[2] # activation 3

    return sigmoid(activation_4)

def predict(row, weights):
    input_layer = row
    first_layer = firstLayer(input_layer,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer, first_layer

def train_weights(train, learning_rate,epochs):
    last_error = 0.0

    for epoch in range(epochs):
        sum_error = 0.0

        for row in train:
            prediction, first_layer = predict(row,weights)
            error = row[-1] - prediction
            sum_error += error

            # First layer out-features
            weights[0] = weights[0] + learning_rate*error
            weights[3] = weights[3] + learning_rate*error
            weights[6] = weights[6] + learning_rate*error
            weights[10] = weights[10] + learning_rate*error

            # First layer in-features
            weights[1] = weights[1] + learning_rate*error*row[0] # activation 1 - mcg
            weights[2] = weights[2] + learning_rate*error*row[1] # activation 1 - gvh

            weights[4] = weights[4] + learning_rate*error*row[2] # activation 2 - lip
            weights[5] = weights[5] + learning_rate*error*row[3] # activation 2 - chg

            weights[7] = weights[7] + learning_rate*error*row[4] # activation 3 - aac
            weights[8] = weights[8] + learning_rate*error*row[5] # activation 3 - alm1
            weights[9] = weights[9] + learning_rate*error*row[6] # activation 3 - alm2

            # Second layer
            weights[11] = weights[11] + learning_rate*error*first_layer[0] # activation 4
            weights[12] = weights[12] + learning_rate*error*first_layer[1] # activation 5
            weights[13] = weights[13] + learning_rate*error*first_layer[2] # activation 6

        if((epoch%100==99) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learning_rate) + " Error " + str(sum_error))
        
        last_error = sum_error
        
    return weights

learningrate = 0.0001
epochs = 10000
train_weights = train_weights(training_dataset, learningrate, epochs)

X = [row[:-1] for row in validation_dataset]
Y = [row[-1] for row in validation_dataset]

predictions = [predict(row, weights) for row in X]

correct_predictions = 0

for i, prediction in enumerate(predictions):
    prediction = round(prediction[0])

    if prediction == Y[i]:
        correct_predictions += 1
    
print("Accuracy:", correct_predictions / len(predictions))