# Importing the libraries

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:73].values
y = dataset.iloc[:, -1].values

# Taking care of the missing values
imputer = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
imputer = imputer.fit(X[:, 0:72])
X[:, 0:72] = imputer.transform(X[:, 0:72])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Fitting XGBoost to the Training set
classifier = XGBClassifier(
    learning_rate=0.1, max_depth=4, min_samples_split=100)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(
    estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
"""
from sklearn.model_selection import GridSearchCV
parameters = [
    {'learning_rate': [0.1, 0.05, 0.02, 0.01], 'max_depth': [1, 2, 3, 4, 6, 8],}
]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

best_parameters
"""

accuracy1 = accuracies.mean()
accuracy2 = accuracies.std()


# Fitting XGBoost to the Training set
classifier = XGBClassifier(
    learning_rate=0.1, max_depth=4, min_samples_split=100)
classifier.fit(X, y)

# Predicting the againist entire set
pred_train = classifier.predict(X)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10, n_jobs=1)

accuracy3 = accuracies.mean()
accuracy4 = accuracies.std()

# Predicting the test set

test_dataset = pd.read_csv('test.csv')
test_X = test_dataset.iloc[:, 1:73].values

# Taking care of the missing values
imputer = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
imputer = imputer.fit(test_X[:, 0:73])
test_X[:, 0:72] = imputer.transform(test_X[:, 0:72])

test_pred = classifier.predict(test_X)

test_dataset['depression_predict'] = test_pred

test_submit = test_dataset[['surveyid', 'depression_predict']]
test_submit.to_csv('prediction.csv', encoding='utf-8', index=False)
