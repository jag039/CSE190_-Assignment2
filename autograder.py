import csv
import random
import sklearn
from sklearn import linear_model
import math
import platform

import submission

import warnings
warnings.filterwarnings("ignore")

performance = {}

print("Python version = " + str(platform.python_version()))
print("sklearn version = " + str(sklearn.__version__))

f = open("default.csv")
reader = csv.reader(f, delimiter=',')

header = next(reader)
header = next(reader) # This file has two headers

# Read the data and extract label and sensitive attributes
dataAttributeLabel = []
for row in reader:
    d = dict(zip(header,row))
    label = d['default payment next month'] == "1"
    attribute = d['EDUCATION'] == "1"
    del d['default payment next month']
    del d['EDUCATION']
    dataAttributeLabel.append((d,attribute,label))


# Compute and print some overall proportions
positive = [l for _,_,l in dataAttributeLabel]
protected = [z for _,z,_ in dataAttributeLabel]
posProtected = [l for _,z,l in dataAttributeLabel if z]
posNonProtected = [l for _,z,l in dataAttributeLabel if not z]

print("Proportion positive = " + str(sum(positive) / len(positive)))
print("Proportion w/ z=1 = " + str(sum(protected) / len(protected)))
print("Proportion positive w/ z=1 = " + str(sum(posProtected) / len(posProtected)))
print("Proportion positive w/ z=0 = " + str(sum(posNonProtected) / len(posNonProtected)))

# Shuffle the dataset
random.seed(0)
random.shuffle(dataAttributeLabel)

# Training/test splits
dataTrain = dataAttributeLabel[:(len(dataAttributeLabel)*5)//10]
dataTest = dataAttributeLabel[(len(dataAttributeLabel)*5)//10:]

# Balanced Error Rate
def BalancedAccuracy(test_predictions,dataTest):
    # d[2] = label
    TP = sum([d[2] and p for (p,d) in zip(test_predictions,dataTest)])
    TN = sum([not d[2] and not p for (p,d) in zip(test_predictions,dataTest)])
    FP = sum([not d[2] and p for (p,d) in zip(test_predictions,dataTest)])
    FN = sum([d[2] and not p for (p,d) in zip(test_predictions,dataTest)])
    # Convert to rates
    TPR = 0
    TNR = 0
    if TP > 0:
        TPR = TP / (TP + FN)
    if TN > 0:
        TNR = TN / (TN + FP)
    return (TPR + TNR) / 2

# Per-group True-Positive Rates (see: demographic parity)
def groupRates(test_predictions,dataTest):
    # True positives for z=1 and z=0
    # d[1] = sensitive attribute; d[2] = label
    TP_z1 = sum([d[2] and p for (p,d) in zip(test_predictions,dataTest) if d[1]])
    FN_z1 = sum([d[2] and not p for (p,d) in zip(test_predictions,dataTest) if d[1]])
    TP_z0 = sum([d[2] and p for (p,d) in zip(test_predictions,dataTest) if not d[1]])
    FN_z0 = sum([d[2] and not p for (p,d) in zip(test_predictions,dataTest) if not d[1]])
    # Convert to rates
    TPR1 = 0
    TPR0 = 0
    if TP_z1 > 0:
        TPR1 = TP_z1 / (TP_z1 + FN_z1)
    if TP_z0 > 0:
        TPR0 = TP_z0 / (TP_z0 + FN_z0)
    return TPR1, TPR0

def metric(acc, rates, trivial):
    TPR1, TPR0 = rates
    score = 0
    if acc < trivial*0.95:
        print("No score assigned: accuracy (" + str(acc) + ") needs to be greater than 0.95*" + str(trivial))
    else:
        score = math.fabs(TPR1 - TPR0)

# Trivial model

def p0feat(d,z):
    # Just concatenate all features together
    return [float(v) for v in d.values()]

X_train0 = [p0feat(d,z) for d,z,_ in dataTrain]
y_train0 = [l for _,_,l in dataTrain]

X_test0 = [p0feat(d,z) for d,z,_ in dataTest]

mod0 = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod0.fit(X_train0, y_train0)

test_predictions0 = mod0.predict(X_test0)

acc0 = BalancedAccuracy(test_predictions0, dataTest)
rates0 = groupRates(test_predictions0, dataTest)

print()
print("Trivial model accuracy = " + str(acc0))
print("Per-group TPRs = " + str(rates0))

results = {}

#######################
# Part 1: Accuracy    #
#######################

# No constraints on features used (excluding of course the label)

X_train1 = [submission.p1feat(d,z) for d,z,_ in dataTrain]
y_train1 = [l for _,_,l in dataTrain]

X_test1 = [submission.p1feat(d,z) for d,z,_ in dataTest]

mod1 = submission.p1model()
mod1.fit(X_train1, y_train1)

test_predictions1 = mod1.predict(X_test1)
test_scores1 = [x[1] for x in mod1.predict_proba(X_test1)] # Probability of positive label for each instance

acc1 = BalancedAccuracy(test_predictions1, dataTest)
rates1 = groupRates(test_predictions1, dataTest)

print()
print("Part 1 accuracy = " + str(acc1))
print("Per-group TPRs = " + str(rates1))

results["Part 1"] = (acc1, rates1)

#########################################
# Part 2: Dataset-based intervention    #
#########################################

# Use the same model from Part 1, but modify the dataset however you like

yourUpdatedData = submission.p2data(dataTrain)

X_train2 = [submission.p1feat(d,z) for d,z,_ in yourUpdatedData]
y_train2 = [l for _,_,l in yourUpdatedData]

mod2 = submission.p2model()
mod2.fit(X_train2, y_train2)

test_predictions2 = mod2.predict(X_test1)

acc2 = BalancedAccuracy(test_predictions2, dataTest)
rates2 = groupRates(test_predictions2, dataTest)

print()
print("Part 2 accuracy = " + str(acc2))
print("Per-group TPRs = " + str(rates2))

results["Part 2"] = (acc2, rates2)

#########################################
# Problem 3: Model-based intervention   #
#########################################

# You can use the sensitive attribute when building your model, but you *cannot* use it at test time

mod = submission.p3model(dataTrain)

X_test3 = [submission.p3feat(d) for d,_,_ in dataTest]
test_predictions3 = mod.predict(X_test3)

acc3 = BalancedAccuracy(test_predictions3, dataTest)
rates3 = groupRates(test_predictions3, dataTest)

print()
print("Part 3 accuracy = " + str(acc3))
print("Per-group TPRs = " + str(rates3))

results["Part 3"] = (acc3, rates3)

###########################################
# Problem 4: Post-processing intervention #
###########################################

# You can change your predictions using the sensitive attribute at test time

test_predictions4 = submission.p4labels(test_scores1, [d for d,_,_ in dataTest], [z for _,z,_ in dataTest])

acc4 = BalancedAccuracy(test_predictions4, dataTest)
rates4 = groupRates(test_predictions4, dataTest)

print()
print("Part 4 accuracy = " + str(acc4))
print("Per-group TPRs = " + str(rates4))

results["Part 4"] = (acc4, rates4)

########################################################
# Problem 5: Any combination of interventions you like #
########################################################

test_predictions5 = submission.p5(dataTrain, [d for d,_,_ in dataTest], [z for _,z,_ in dataTest])

acc5 = BalancedAccuracy(test_predictions5, dataTest)
rates5 = groupRates(test_predictions5, dataTest)

print()
print("Part 5 accuracy = " + str(acc5))
print("Per-group TPRs = " + str(rates5))

results["Part 5"] = (acc5, rates5)

# "results" will be used by the autograder to compile your final score
