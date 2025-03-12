#!/usr/bin/env python
# coding: utf-8

# ## HW 3: Fairness and Bias Interventions

# ## Download the dataset
# 
# 1. Go to the [Adult Dataset webpage](https://archive.ics.uci.edu/dataset/2/adult).
# 2. Download and unzip the file in the same directory as this notebook.

# In[1]:


# TODO: download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math

import numpy as np

# In[2]:


## DATA LOADING ##
header = ["age",
          "workclass",
          "fnlwgt",
          "education",
          "education-num",
          "marital-status",
          "occupation",
          "relationship",
          "race",
          "sex",
          "capital-gain",
          "capital-loss",
          "hours-per-week",
          "native-country"]

values = {"workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
          "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
          "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
          "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
          "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
          "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
          "sex": ["Female", "Male"],
          "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
         }

# In[3]:


def feat(d):
    f = [1]
    for h in header:
        if h in values:
            onehot = [0]*len(values[h])
            try:
                onehot[values[h].index(d[h])] = 1 # not efficient! Should make an index
            except Exception as e:
                # Missing value
                pass
            f += onehot
        else: # continuous
            try:
                f.append(float(d[h]))
            except Exception as e:
                # Missing value
                f.append(0) # Replacing with zero probably not perfect!
    return f

# In[4]:


dataset = []
labels = []
a = open("/home/dpach/Documents/cse190/hw3/adult/adult.data", 'r')
for l in a:
    if len(l) <= 1: break # Last line of the dataset is empty
    l = l.split(", ") # Could use a csv library but probably no need to here
    dataset.append(dict(zip(header, l)))
    labels.append(l[-1].strip()) # Last entry in each row is the label

X = [feat(d) for d in dataset]
y = [inc == '>50K' for inc in labels]
print(dataset[:10])
X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(X, y, dataset, test_size=0.2, random_state=42)

# In[5]:


d_train

# In[6]:


answers = {}

# ## 3.1
# 
# #### (1 point)
# 
# Implement a logistic regression classification pipeline using an `80/20` test split. Use a regularization value of $C = 1$.
# 
# Treat “sex” as the “sensitive attribute” i.e., $z=1$ for females and $z=0$ for others.
# 
# **Report:** The discrimination in the dataset (see "pre-processing" module).

# In[7]:





def discrimination_score(datapoints, labels):
    count_female = 0
    count_male = 0
    for d in datapoints:
        if d["sex"] == "Female":
            count_female += 1
        if d["sex"] == "Male":
            count_male += 1
    count_male_positive = 0
    count_female_positive = 0
    for i in range(len(datapoints)):
        if labels[i] == True and datapoints[i]["sex"] == "Female":
            count_female_positive += 1
        if labels[i] == True and datapoints[i]["sex"] == "Male":
            count_male_positive += 1
    term1 = count_male_positive/count_male
    term2 = count_female_positive/count_female

    return abs(term1 - term2)


dataset_discrimination = discrimination_score(d_train, y_train)
print(f'Dataset discrimination: {dataset_discrimination:.6f}')

# In[8]:


answers['Q1'] = dataset_discrimination

# ## 3.2
# 
# #### (1 point)
# 
# **Report:** The discrimination of the classifier.

# In[9]:




model_q2 = LogisticRegression(C=1)
preds_train_q2 = model_q2.fit(X_train, y_train).predict(X_train)
classifier_discrimination_q2 = discrimination_score(d_train, preds_train_q2)

print(f'Classifier discrimination (Q2): {classifier_discrimination_q2:.6f}')

# In[10]:


answers['Q2'] = classifier_discrimination_q2

# In[11]:


# need disc score of 0.0437

# In[12]:


scorestrain = model_q2.decision_function(X_train)

# In[13]:


pr = [(sc,ind) for (sc,ind,d,y_) in zip(scorestrain,range(len(d_train)),d_train,y_train) if d['sex'] == "Female" and not y_]
pr.sort(reverse=True)


# In[14]:


dem = [(sc,ind) for (sc,ind,d,y_) in zip(scorestrain,range(len(d_train)),d_train,y_train) if d['sex'] == "Male" and y_]
dem.sort()

# In[15]:




desc_d = dataset_discrimination 
d1 = np.sum([1 for d in d_train if d["sex"] == "Female"])
d0 = len(d_train) - d1
M = (desc_d * d1 * d0)/len(d_train)
M = math.ceil(M)
y_train_fixed_q3 = y_train[:]
for i in range(M):
    y_train_fixed_q3[pr[i][1]] = True
    y_train_fixed_q3[dem[i][1]] = False
# train model
model_q3 = LogisticRegression(C=1)
preds_train_q3 = model_q3.fit(X_train, y_train_fixed_q3).predict(X_train)
classifier_discrimination_q3 = discrimination_score(d_train, preds_train_q3)

print(f'Classifier discrimination (Q3):\t {classifier_discrimination_q3:.6f}')
print(f'Classifier relative improvement (Q3):\t {100*(classifier_discrimination_q2 - classifier_discrimination_q3) / classifier_discrimination_q2:.6f}%')

# In[16]:


answers['Q3'] = classifier_discrimination_q3

# ## 3.4
# 
# #### (2 points)
# 
# Implement a "reweighting" approach that improves the discrimination score by at least 3%; report the new discrimination score.
# 
# **Report:** The new discrimination score.

# In[17]:



n_female = np.sum([1 for d in d_train if d["sex"] == "Female"])
n_male = len(d_train) - n_female

n_pos = np.sum(y_train)
n_neg = len(y_train) - n_pos

n_pos_female = np.sum([1 for d, y in zip(d_train, y_train) if d["sex"] == "Female" and y == 1])
n_neg_female = np.sum([1 for d, y in zip(d_train, y_train) if d["sex"] == "Female" and y == 0])

n_pos_male = np.sum([1 for d, y in zip(d_train, y_train) if d["sex"] == "Male" and y == 1])
n_neg_male = np.sum([1 for d, y in zip(d_train, y_train) if d["sex"] == "Male" and y == 0])

# 
w_pos_female = (n_female * n_pos) / (len(d_train) * n_pos_female)
w_neg_female = (n_female * n_neg) / (len(d_train) * n_neg_female)

w_pos_male = (n_male * n_pos) / (len(d_train) * n_pos_male)
w_neg_male = (n_male * n_neg) / (len(d_train) * n_neg_male)

weights = np.array([w_pos_female if d["sex"] == "Female" and y == 1 else
                    w_neg_female if d["sex"] == "Female" and y == 0 else
                    w_pos_male if d["sex"] == "Male" and y == 1 else
                    w_neg_male for d, y in zip(d_train, y_train)])

model_q4 = LogisticRegression(C=1)
preds_train_q4 = model_q4.fit(X_train, y_train, sample_weight=weights).predict(X_train)
classifier_discrimination_q4 = discrimination_score(d_train, preds_train_q4)

print(f'Classifier discrimination (Q4):\t {classifier_discrimination_q4:.6f}')
print(f'Classifier relative improvement (Q4):\t {100*(classifier_discrimination_q2 - classifier_discrimination_q4) / classifier_discrimination_q2:.6f}%')

# In[18]:


answers['Q4'] = classifier_discrimination_q4

# ## 3.5
# 
# #### (2 points)
# 
# Implement a "post processing" (affirmative action) policy. Lowering per-group thresholds will increase both the (per-group) FPR and the (per-group) TPR. For whichever group has the lower TPR, lower the threshold until the TPR for both groups is (as close as possible to) equal. Report the rates (TPR_0, TPR_1, FPR_0, and FPR_1) for both groups.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.
# 

# In[19]:


def find_rates(labels, scores, thr):
    pred_label = [s > thr for s in scores]
    TP = sum([1 for i, j in zip(labels, pred_label) if i and j])
    TN = sum([1 for i,j in zip(labels, pred_label) if not i and not j])
    FP = sum([1 for i, j in zip(labels, pred_label) if not i and j])
    FN = sum([1 for i, j in zip(labels, pred_label) if i and not j])
    if TP + FN == 0:
        TPR = 0
    else:
        TPR = TP / (TP + FN)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)

    return TPR, FPR

# In[20]:


scores = model_q2.predict_proba(X_train)[:, 1] # proba of class being 1


# In[21]:


male = [s for s, d in zip(scores, d_train) if d["sex"] == "Male"]
female = [s for s, d in zip(scores, d_train) if d["sex"]== "Female"]

male_labels = [y for y, d in zip(y_train, d_train) if d["sex"] == "Male"]
female_labels = [y for y, d in zip(y_train, d_train) if d["sex"] == "Female"]


# In[22]:


male_thresh = 0.5
female_thresh_q5 = 0.5


# In[23]:


TPR_male, FPR_male = find_rates(male_labels, male, male_thresh)
TPR_female, FPR_female = find_rates(female_labels, female, female_thresh_q5)
TPR_female, TPR_male, FPR_male, FPR_female

# In[24]:


if TPR_male < TPR_female:
    for thr in np.linspace(0.5, 0, 100):
        TPR_male_new, FPR_male_new = find_rates(male_labels, male, thr)
        if TPR_male_new >= TPR_female:
            male_thresh = thr
            print(f"Male: {male_thresh}")
            TPR_male, FPR_male = TPR_male_new, FPR_male_new
            break
elif TPR_female < TPR_male:
    for thr in np.linspace(0.5, 0, 100):
        TPR_female_new, FPR_female_new = find_rates(female_labels, female, thr)
        if TPR_female_new >= TPR_male:
            female_thresh_q5 = thr
            print(female_thresh_q5)
            TPR_female, FPR_female = TPR_female_new, FPR_female_new
            break

# In[25]:


ans_q5 = [TPR_male, TPR_female, FPR_male, FPR_female]  # [TPR_male, TPR_female, FPR_male, FPR_female]

print(ans_q5)

# In[26]:


answers['Q5'] = ans_q5

# ## 3.6
# 
# #### (1 point)
# 
# Modify the solution from Q5 to exclude the sensitive attribute ($z$) from the classifier’s feature vector. Implement the same strategy as in Q5.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.
# 

# In[27]:


sensitive_attribute = 9

# In[28]:


# Remove the "sex" attribute from the feature vector
X_train_no_sex = [{k: v for k, v in d.items() if k != "sex"} for d in d_train]

# In[29]:


# Convert the modified dataset into feature vectors
model_q6 = LogisticRegression(C=1)
model_q6.fit(X_train, y_train)

# In[30]:


scores_q6 = model_q6.predict_proba(X_train)[:, 1] # proba of class being 1


# In[31]:


male = [s for s, d in zip(scores_q6, d_train) if d["sex"] == "Male"]
female = [s for s, d in zip(scores_q6, d_train) if d["sex"]== "Female"]

male_labels = [y for y, d in zip(y_train, d_train) if d["sex"] == "Male"]
female_labels = [y for y, d in zip(y_train, d_train) if d["sex"] == "Female"]


# In[32]:


male_thresh = 0.5
female_thresh_q6 = 0.5


# In[33]:


TPR_male, FPR_male = find_rates(male_labels, male, male_thresh)
TPR_female, FPR_female = find_rates(female_labels, female, female_thresh_q6)
TPR_female, TPR_male, FPR_male, FPR_female

# In[34]:


if TPR_male < TPR_female:
    for thr in np.linspace(0.5, 0, 100):
        TPR_male_new, FPR_male_new = find_rates(male_labels, male, thr)
        if TPR_male_new >= TPR_female:
            male_thresh = thr
            print(f"Male: {male_thresh}")
            TPR_male, FPR_male = TPR_male_new, FPR_male_new
            break
elif TPR_female < TPR_male:
    for thr in np.linspace(0.5, 0, 100):
        TPR_female_new, FPR_female_new = find_rates(female_labels, female, thr)
        if TPR_female_new >= TPR_male:
            female_thresh_q5 = thr
            print(female_thresh_q5)
            TPR_female, FPR_female = TPR_female_new, FPR_female_new
            break

# In[35]:


ans_q6 = [TPR_male, TPR_female, FPR_male, FPR_female]

print(ans_q6)

# In[36]:


answers['Q6'] = ans_q6

# ## 3.7
# 
# #### (1 point)
# 
# Again modifying the solution from Q5, train two separate classifiers, one for $z=0$ and one for $z=1$. Implement the same strategy as in Q5.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.

# In[37]:


X_train_male = np.array([x for x, d in zip(X_train, d_train) if d["sex"] == "Male"])
X_train_female = np.array([x for x, d in zip(X_train, d_train) if d["sex"] == "Female"])

y_train_male = np.array([y for y, d in zip(y_train, d_train) if d["sex"] == "Male"])
y_train_female = np.array([y for y, d in zip(y_train, d_train) if d["sex"] == "Female"])


# In[38]:



model_male = LogisticRegression(C=1)
model_male.fit(X_train_male, y_train_male)

model_female = LogisticRegression(C=1)
model_female.fit(X_train_female, y_train_female)

male_scores = model_male.predict_proba(X_train_male)[:, 1]
femlae_scores = model_female.predict_proba(X_train_female)[:, 1]




# In[39]:


male_thresh = 0.5
female_thresh_q7 = 0.5

# In[40]:


TPR_male, FPR_male = find_rates(y_train_male, male_scores, male_thresh)
TPR_female, FPR_female = find_rates(y_train_female, femlae_scores, female_thresh_q7)

# In[41]:


if TPR_male < TPR_female:
    for thr in np.linspace(male_thresh, 0, 100):
        TPR_male_new, FPR_male_new = find_rates(y_train_male, male_scores, thr)
        if TPR_male_new >= TPR_female:
            male_thresh = thr
            TPR_male, FPR_male = TPR_male_new, FPR_male_new
            break
elif TPR_female < TPR_male:
    for thr in np.linspace(female_thresh_q7, 0, 100):
        TPR_female_new, FPR_female_new = find_rates(y_train_female, femlae_scores, thr)
        if TPR_female_new >= TPR_male:
            female_thresh_q7 = thr
            TPR_female, FPR_female = TPR_female_new, FPR_female_new
            break

ans_q7 = [TPR_male, TPR_female, FPR_male, FPR_female]


# In[44]:


answers['Q7'] = ans_q7

# ## Saving Answers

# In[45]:


import json

# ## 3.3
# #### (1 point)
# 
# Implement a "massaging" approach that improves the discrimination score by at least 3\%.
# 
# 
# **Report:** The new discrimination score.

# In[46]:


# extra step to make things serializable

with open('answers_hw3.txt', 'w' ) as f:
    json.dump(answers, f, indent=2)

# In[ ]:



