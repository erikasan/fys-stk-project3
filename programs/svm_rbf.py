import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


gammas = np.logspace(-4, -0.5, 150)
scores = np.zeros(gammas.shape)
std = np.zeros(gammas.shape)

for i, gamma in enumerate(gammas):
    rbf_clf = SVC(gamma=gamma, kernel='rbf')
    score      = cross_val_score(rbf_clf, X, y, cv=5)
    scores[i]  = score.mean()
    std[i]     = score.std()
    print(f'{i+1}/{gammas.size} finished')

svm_rbf = np.vstack((gammas, scores, std)).T
np.save('svm_rbf', svm_rbf)
