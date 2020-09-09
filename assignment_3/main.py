import csv,math
import json

# split the data
def split(data, index):
    return_values = {}

    for row in data:
        value = row[index]
        rows = return_values.get(value, [])
        rows.append(row)
        return_values[value] = rows

    return return_values


# calculate the entropy
def entropy(rows):
    negative = len([row for row in rows if row[0] =="0"])
    positive = len([row for row in rows if row[0] =="1"])

    total = negative + positive

    # if all rows are either positive or negative, the entropy is 0
    if min(positive, negative) == 0:
        return 0
    
    # standard formula for calculating the entropy
    return -(positive / total) * math.log(positive / total, 2) - (negative / total) * math.log(negative / total, 2)


# calculate the gain for a single attribute
def gain(rows, index):
    entropies = [(entropy(row), len(row)) for row in split(rows, index).values()]
    total_rows = sum(entropy[1] for entropy in entropies)
    return sum((entropy[0] * entropy[1]) / total_rows for entropy in entropies)


# find the attribute that yields the highest gain for a given row
def get_highest_gain(rows):
    indices = [i for i in range(1, len(rows[0]))] # all indices except for recurrent / non-recurrent
    gains = [gain(rows, index) for index in indices]
    return gains.index(min(gains)) + 1 # we want the index of the attribute, not the value


# checks if the dataset is pure
def is_pure(rows):
    indices = [i for i in range(1, len(rows[0]))] # all indices except for recurrent / non-recurrent

    for index in indices:
        unique_attributes = set([row[index] for row in rows])

        if len(unique_attributes) > 1:
            return False
        
    return True


# returns the most common of 'non-recurrent' or 'recurrent'
def most_common(rows):
    recurrence_results = [row[0] for row in rows]
    return max(set(recurrence_results), key=recurrence_results.count)


def is_empty(rows):
    return len(rows) <= 1

classifier_code = "def classify(data):"

def build_tree(dataset, spaces="    "):
    global classifier_code

    if(is_empty(dataset) or is_pure(dataset)):
        print(spaces, "then", most_common(dataset))
        classifier_code += "\n" + spaces + "return (" + most_common(dataset) + ")"
        return

    highest = get_highest_gain(dataset)
    data_split = split(dataset,highest)

    # if we are unable to make a choice
    if len(data_split) == 1:
        print(spaces + spaces, "then", most_common(dataset))
        classifier_code += "\n" + spaces + "return (" + most_common(dataset) + ")"
        return
    else:
        for key, value in data_split.items():
            print(spaces, "if", key)
            classifier_code += "\n" + spaces + "if(data[" + str(highest) + "] == \"" + str(key) +"\"):"
            build_tree(value, spaces + "   ")


# read in data
rows = csv.reader(open('breast-cancer.data', 'r'))

# convert from csv format
data = [row for row in rows]

for row in data:
    if row[0] == "no-recurrence-events":
        row[0] = "0"
    else:
        row[0] = "1"

# split into training data and verification
training_data = data[int(len(data) / 2):]
verification_data = data[:int(len(data) / 2)]

build_tree(training_data)

classifier_code += "\n    return (0)"
print(classifier_code)

exec(classifier_code)
#print(classify(verification_data[0]),verification_data[0])

correct,wrong = 0,0
for d in verification_data:
    if(int(d[0])==int(classify(d))):
        correct += 1
    else:
        wrong += 1
print("Correct classifications",correct)
print("Wrong classifications",wrong)
print("Accuracy",(correct/(correct+wrong)))
