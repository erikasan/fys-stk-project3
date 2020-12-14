import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from true_false import true_false_map

# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
# print(X_train.shape)
# print(X_test.shape)

# Logistic Regression
logreg = LogisticRegression(solver='sag', max_iter=5000)
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression: {:.3f}".format(logreg.score(X_test,y_test)))

predictions = logreg.predict(X_test)


true_false_map(predictions, y_test, 'Logistic Regression')




scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
logreg = LogisticRegression(solver='sag', max_iter=500)
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy with Logistic Regression scaled: {:.3f}".format(logreg.score(X_test_scaled,y_test)))

predictions = logreg.predict(X_test_scaled)


#true_false_map(predictions, y_test, 'Logistic Regression scaled')
