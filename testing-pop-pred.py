import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


data = pd.read_csv("trimmed_pop_data.csv")

## create workable dictionary ##
frame = {}
for i in range(0, len(data['Country'])):   ## build dict: frame containing a dict for every country which will hold all features as keyed lists
    row = data.iloc[[i]].drop('Country', axis = 1)
    frame[data['Country'][i]] = {
        'pop':row
    }

## build larger set of all possible length-10 pop periods with 11th year as label ##
examples = []
labels = []
for key in frame:
    poplist = frame[key]['pop'].values.flatten().tolist()
    s = 0
    f = 10
    while f < 57:
        examples.append(poplist[s:f])
        labels.append(poplist[f])
        s += 1
        f += 1
examples = np.array(examples)
labels = np.array(labels)
## remove examples containing or labeled 'Nan' ##

# remove examples containing nan & their associated labels
def twodpull(X, y):
    nanInds = []
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            if np.isnan(X[i][j]):
                nanInds.append(i)
                break
    y = np.delete(y, nanInds) 
    X = np.delete(X, nanInds, 0) # new set of examples has no nans
    return X, y

# remove nan labels & their associated examples
def pull(y, X):      # input: y: list containing labels; X: list of lists (examples)       
    nanInds = []        # to store inds of nan labels
    for i in range(0, len(y)):
        if np.isnan(y[i]):
            nanInds.append(i)
    y = np.delete(y, nanInds) # new list of labels has no nans
    X = np.delete(X, nanInds, 0) # new set of examples only have real-val labels
    return y, X # output: (newy, newX)

examples, labels = twodpull(examples, labels)
labels, examples = pull(labels, examples)

train = examples
target = labels

train, target = shuffle(train, target, random_state=10)

training = {
    'X': np.array(train[:10000]), # use np.array for feeding to tensorflow eventually
    'Y': np.array(target[:10000])
}

validation = {
    'X': np.array(train[10000:]),
    'Y': np.array(target[10000:])
}

X_train = training['X'] #.astype(np.float32)
y_train = training['Y'] #.astype(np.float32)
X_test = validation['X'] #.astype(np.float32)
y_test = validation['Y'] #.astype(np.float32)

params = {'n_estimators': 500, 'max_depth': 2, 'random_state': 10,
          'learning_rate': 0.01, 'loss': 'ls'}
est = ensemble.GradientBoostingRegressor(**params)

est.fit(X_train, y_train)
trainpreds = est.predict(X_train)
trmse = mean_squared_error(y_train, trainpreds)
testpreds = est.predict(X_test)
mse = mean_squared_error(y_test, testpreds)

print("MSE: %.4f" % mse)
print("TRMSE: %.4f" % trmse)
ratio = mse/trmse
print("Ratio test mse:training mse %.4f" % ratio)





