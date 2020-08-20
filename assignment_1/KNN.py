import numpy
import statistics
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
print(X_train)
print(Y_train)

def distance(one,two):
    return numpy.linalg.norm(one-two)

def predict(x,x_rest,y_rest):
    neighbors = [(distance(x, x_rest[i]), y_rest[i]) for i in range(len(x_rest))]
    neighbors.sort(key=lambda x: x[0])

    prediction = statistics.mode([neighbor[1] for neighbor in neighbors[0:5]])
    
    return prediction


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]

    pred = predict(x,X_train,Y_train)

    if(y==1 and pred ==1):
        TP += 1

    if(y==0 and pred ==0):
        TN += 1

    if(y==1 and pred ==0):
        FN += 1

    if(y==0 and pred ==1):
        FP += 1

print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
print("Recall",TP/(TP+FN))
print("Precision",TP/(TP+FP))
print("F1",(2*TP)/(2*TP+FP+FN))

