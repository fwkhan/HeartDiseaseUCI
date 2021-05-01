import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

data = pd.read_csv('Data/heart.csv')
data.head()


# extract the data

def extract_data(dataset):
    # extract columns 'age'
    dataset.loc[dataset['age'] <= 40, 'age'] = 0
    dataset.loc[(dataset['age'] > 40) & (dataset['age'] <= 50), 'age'] = 1
    dataset.loc[(dataset['age'] > 50) & (dataset['age'] <= 60), 'age'] = 2
    dataset.loc[(dataset['age'] > 60) & (dataset['age'] <= 70), 'age'] = 3
    dataset.loc[dataset['age'] > 70, 'age'] = 4

    # extract columns 'trestbps'
    dataset.loc[dataset['trestbps'] <= 116, 'trestbps'] = 0
    dataset.loc[(dataset['trestbps'] > 116) & (dataset['trestbps'] <= 138), 'trestbps'] = 1
    dataset.loc[(dataset['trestbps'] > 138) & (dataset['trestbps'] <= 150), 'trestbps'] = 2
    dataset.loc[(dataset['trestbps'] > 150) & (dataset['trestbps'] <= 172), 'trestbps'] = 3
    dataset.loc[dataset['trestbps'] > 172, 'trestbps'] = 4

    # extract columns 'chol'
    dataset.loc[dataset['chol'] <= 214, 'chol'] = 0
    dataset.loc[(dataset['chol'] > 214) & (dataset['chol'] <= 302), 'chol'] = 1
    dataset.loc[(dataset['chol'] > 302) & (dataset['chol'] <= 390), 'chol'] = 2
    dataset.loc[(dataset['chol'] > 390) & (dataset['chol'] <= 478), 'chol'] = 3
    dataset.loc[dataset['chol'] > 478, 'chol'] = 4

    # extract columns 'thalach'
    dataset.loc[dataset['thalach'] <= 98, 'thalach'] = 0
    dataset.loc[(dataset['thalach'] > 98) & (dataset['thalach'] <= 125), 'thalach'] = 1
    dataset.loc[(dataset['thalach'] > 125) & (dataset['thalach'] <= 152), 'thalach'] = 2
    dataset.loc[(dataset['thalach'] > 152) & (dataset['thalach'] <= 179), 'thalach'] = 3
    dataset.loc[dataset['thalach'] > 179, 'thalach'] = 4

    return dataset


extract_data(data)
data.head()
print(data)

from sklearn.model_selection import train_test_split

x = data.iloc[:, :-2]
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


def pickle_models(model, filename):
    path = "Models/"+f'{filename}'
    outfile = open(path, 'wb')
    pickle.dump(model, outfile)
    outfile.close()



def logisticregression(train, test):
    model = LogisticRegression(random_state=0).fit(train, y_train)
    y_train_pred = model.predict(train)
    y_test_pred = model.predict(test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    matrix = confusion_matrix(y_test, y_test_pred)
    print('Logistic Regression')
    print(matrix)
    print('Train accuracy is', train_score)
    print('Test accuracy is', test_score)
    pickle_models(model, "LogisticRegression")


def decisiontree(train, test):
    model = tree.DecisionTreeClassifier(criterion='entropy')
    model = model.fit(train, y_train)
    y_train_pred = model.predict(train)
    y_test_pred = model.predict(test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    matrix = confusion_matrix(y_test, y_test_pred)
    print(' ')
    print('Decision Tree')
    print(matrix)
    print('Train accuracy is', train_score)
    print('Test accuracy is', test_score)
    pickle_models(model, "DecisionTree")


def naivebays(train, test):
    model = GaussianNB()
    model = model.fit(train, y_train)
    y_train_pred = model.predict(train)
    y_test_pred = model.predict(test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    matrix = confusion_matrix(y_test, y_test_pred)
    print(' ')
    print('Native Bayes')
    print(matrix)
    print('Train accuracy is', train_score)
    print('Test accuracy is', test_score)
    pickle_models(model, "NaiveBayes")


logisticregression(x_train, x_test)
decisiontree(x_train, x_test)
naivebays(x_train, x_test)
print(x_test)
