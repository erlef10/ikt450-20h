import csv, math

data = csv.reader(open('breast-cancer.data', 'r'))
print(data)
all_data = [d for d in data]

for d in all_data:
    if d[0] == 'no-recurrence-events':
        d[0] = '0'
    else:
        d[0] = '1'

print(all_data)

all_data = all_data[1:]
trainingdata = all_data[int(len(all_data)/2):]
verificationdata = all_data[:int(len(all_data)/2)]

def entropy(oneclass):
    pos = len([i for i in oneclass if i[0] == "0"])
    neg = len([i for i in oneclass if i[0] == "1"])
    total = pos+neg
    if(min(pos,neg)==0):
        return 0
    entropy = - (pos/total)*math.log(pos/total,2) - (neg/total)*math.log(neg/total,2)
    return entropy

print(entropy(trainingdata))

def split(data,attribute,remove=False):
    #import pdb;pdb.set_trace()
    retvals = {}
    allattributes = set([i[attribute] for i in data])
    for d in data:
        c = d[attribute]
        aList = retvals.get(c, [])
        if remove:
            d.pop(attribute)
        aList.append(d)
        retvals[c] = aList
    return retvals


def gain(oneclass, attribute):
    d = [(entropy(i), len(i)) for i in split(oneclass, attribute).values()]
    nAll = sum(i[1] for i in d)
    gain = sum([(i[0]*i[1])/nAll for i in d])
    return gain

def getHighestGain(oneclass):
    classes = [i for i in range(1, len(oneclass[0]))]
    entropies = [gain(oneclass, c) for c in classes]
    return entropies.index(min(entropies)) + 1

print(getHighestGain(trainingdata))

def isPure(oneclass):
    classes = [i for i in range(1, len(oneclass[0]))]

    for c in classes:
        unique = set([i[c] for i in oneclass])
        if(len(unique) > 1):
            return False
    return True

def isEmpty(oneclass):
    return len(oneclass[0])<=1

def mostCommon(oneclass):
    lst = [i[0] for i in oneclass]
    return max(set(lst), key=lst.count)

actualClassifier = "def classify(data):"

def buildTree(oneclass,spaces="    "):
    global actualClassifier
    if(isEmpty(oneclass) or isPure(oneclass)):
        print(spaces,"then",mostCommon(oneclass))
        #print(spaces,"#confidence",confidence(oneclass))
        actualClassifier += "\n"+spaces+"return ("+mostCommon(oneclass)+")"
        return
    highest = getHighestGain(oneclass)
    d = split(oneclass, highest)
    for key, value in d.items():
        print(spaces,"if",key)
        actualClassifier += "\n"+spaces+"if(data["+str(highest)+"]==\""+str(key)+"\"):"
        buildTree(value, spaces+"   ")

buildTree(trainingdata)
print(actualClassifier)

exec(actualClassifier)
print(classify(verificationdata[0]),verificationdata[0])

correct,wrong = 0,0
for data in verificationdata:
    if(int(data[0])==int(classify(data))):
        correct += 1
    else:
        wrong += 1
print("Correct classifications",correct)
print("Wrong classifications",wrong)
print("Accuracy",(correct/(correct+wrong)))
