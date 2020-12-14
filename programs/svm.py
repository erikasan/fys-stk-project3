import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from true_false import true_false_map

# # Load the data
# X, y = load_breast_cancer(return_X_y=True)
#
# # Split the data into training and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Create a SVM classifier
# clf = SVC()
#
# # Train the classifier
# clf.fit(X_train, y_train)
#
# # Calculate the mean accuracy
# score = clf.score(X_test, y_test)
#
# print(f'score = {score}')


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

C_penalty = np.logspace(-3, 2, 6)
scores = np.zeros(C_penalty.shape)
for i, C in enumerate(C_penalty):
    linear_clf = SVC(C=C, kernel='linear')
    linear_clf.fit(X_train, y_train)
    scores[i] = linear_clf.score(X_test, y_test)

plt.plot(C_penalty, scores, '-o')
plt.xlabel(r'$C$')
plt.ylabel(r'Mean Accuracy')
plt.show()
