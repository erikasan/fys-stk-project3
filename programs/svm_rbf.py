import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer



X, y = load_breast_cancer(return_X_y=True)

gammas = np.logspace(-10, -2, 150)

scores = np.zeros(gammas.shape)
std = np.zeros(gammas.shape)

for i, gamma in enumerate(gammas):
    rbf_clf = SVC(gamma=gamma, kernel='rbf')
    score      = cross_val_score(rbf_clf, X, y, cv=5)
    scores[i]  = score.mean()
    std[i]     = score.std()

svm_rbf = np.vstack((gammas, scores, std)).T
np.save('svm_rbf', svm_rbf)
