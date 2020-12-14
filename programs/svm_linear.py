import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer



X, y = load_breast_cancer(return_X_y=True)

C_penalty = np.logspace(-2, 6, 9)
scores = np.zeros(C_penalty.shape)
std = np.zeros(C_penalty.shape)
for i, C in enumerate(C_penalty):
    linear_clf = SVC(C=C, kernel='linear')
    score      = cross_val_score(linear_clf, X, y, cv=5)
    scores[i]  = score.mean()
    std[i]     = score.std()

svm_linear = np.vstack((C_penalty, scores, std)).T
np.save('svm_linear', svm_linear)
