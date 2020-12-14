import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from true_false import true_false_map


# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
# print(X_train.shape)
# print(X_test.shape)


# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(X_train, y_train)
print("Test set accuracy with Decision Trees: {:.3f}".format(deep_tree_clf.score(X_test,y_test)))

predict_tree = deep_tree_clf.predict(X_test)

true_false_map(predict_tree, y_test, 'Decision Tree')

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(X_train_scaled, y_train)
print("Test set accuracy with Decision Trees scaled: {:.3f}".format(deep_tree_clf.score(X_test_scaled,y_test)))

predict_tree = deep_tree_clf.predict(X_test_scaled)

#true_false_map(predict_tree, y_test, 'Decision Tree scaled')
