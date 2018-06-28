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

usadata = data[246:247]
usapred = usadata[usadata.columns[-10:]]
usapred = np.array(usapred)

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

X_train = training['X'] 
y_train = training['Y'] 
X_test = validation['X'] 
y_test = validation['Y'] 

params = {'n_estimators': 500, 'max_depth': 4, 'random_state': 10,
          'learning_rate': 0.05, 'loss': 'ls'}
est = ensemble.GradientBoostingRegressor(**params)

est.fit(X_train, y_train)
trainpreds = est.predict(X_train)
trmse = mean_squared_error(y_train, trainpreds)
testpreds = est.predict(X_test)
mse = mean_squared_error(y_test, testpreds)
rmse = np.sqrt(mse)
trrmse = np.sqrt(trmse)
print("RMSE: %.4f" % rmse)
print("TrRMSE: %.4f" % trrmse)
ratio = rmse/trrmse
print("Ratio test rmse:training rmse %.4f" % ratio)

USApredictions = []
usalead = usapred
q=0
while q < 10:
    usapred = usapred.reshape(1, -1)
    pred = est.predict(usapred)
    USApredictions = np.append(USApredictions, pred)
    usapred = np.delete(usapred, 0)
    usapred = np.append(usapred, pred)
    q+=1
USApredictions = USApredictions.reshape(1, -1)
print(USApredictions)

plotdata = np.concatenate((usalead, USApredictions), axis = 0)
plotdata = plotdata.flatten()
years = np.arange(2007, 2027)
plt.plot(years, plotdata, 'g*--')
plt.show()
