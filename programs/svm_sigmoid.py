import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


gammas = np.linspace(0.001, 0.06, 100)
scores = np.zeros(gammas.size)
std    = np.zeros(gammas.size)

for i, gamma in enumerate(gammas):
    poly_clf  = SVC(gamma=gamma, kernel='sigmoid')
    score     = cross_val_score(poly_clf, X, y, cv=5)
    scores[i] = score.mean()
    std[i]    = score.std()
    print(f'{i+1}/{gammas.size} finished')

svm_sigmoid = np.vstack((gammas, scores, std)).T
np.save('svm_sigmoid', svm_sigmoid)
