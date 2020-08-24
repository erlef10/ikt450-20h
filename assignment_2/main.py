import numpy as np

# TODO: Learn to manipulate arrays
# import dataset
dataset = np.loadtxt("ecoli.data", delimiter="\t", dtype=str)

rows = []

for i in range(len(dataset)):
    rows.append([x for x in dataset[i].split(" ") if x.strip() != ""])

dataset = rows

rows = []

# remove unecessary rows
for row in dataset:
    if row[-1] not in ["im", "cp"]:
        continue

    rows.append(row)

dataset = rows

# TODO: Check if the first column is unique for every value

# Print the strain of ecoli
#[print(row[-1]) for row in dataset]

# Set 'im' as 1 and 'cp' as 0
training_dataset = [row[:-1]+[0 if row[-1] == 'cp' else 1] for row in dataset]

# remove the first column as they are all unique anyhow
training_dataset = [row[1::] for row in training_dataset]
print(training_dataset[0])

# TODO: Figure out what to connect and how many weights we need
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

import math
def sigmoid(z):
    if(z<-100):
        return 0
    if(z>100):
        return 1
    return 1.0/math.exp(-z)

# TODO: Fix
def firstLayer(row,weights):
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]

    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    return sigmoid(activation_1),sigmoid(activation_2)

# def secondLayer(row,weights):
#     activation_3 = weights[6]
#     activation_3 += weights[7]*row[0]
#     activation_3 += weights[8]*row[1]
#     return sigmoid(activation_3)

# def predict(row,weights):
#     input_layer = row
#     first_layer = firstLayer(input_layer,weights)
#     second_layer = secondLayer(first_layer,weights)
#     return second_layer,first_layer

# # perform predictions
# for d in training_dataset:
#     print(predict(d,weights)[0],d[-1])   #Prints y_hat and y

# def train_weights(train,learningrate,epochs):
#     for epoch in range(epochs):
#         sum_error = 0.0
#         for row in train:
#             prediction,first_layer = predict(row,weights)
#             error = row[-1]-prediction
#             sum_error += error
#             #First layer
#             weights[0] = weights[0] + learningrate*error*1
#             weights[3] = weights[3] + learningrate*error

#             weights[1] = weights[1] + learningrate*error*row[0]
#             weights[2] = weights[2] + learningrate*error*row[1]
#             weights[4] = weights[4] + learningrate*error*row[2]
#             weights[5] = weights[5] + learningrate*error*row[3]

#             #Second layer
#             weights[6] = weights[6] + learningrate*error
#             weights[7] = weights[7] + learningrate*error*first_layer[0]
#             weights[8] = weights[8] + learningrate*error*first_layer[1]
#         if((epoch%100==0) or (last_error != sum_error)):
#             print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
#         last_error = sum_error
#     return weights

# learningrate = 0.0001#0.00001
# epochs = 1000
# train_weights = train_weights(training_dataset,learningrate,epochs)
# print(train_weights)