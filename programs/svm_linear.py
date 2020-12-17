import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)


C_penalty = np.logspace(-4, 5, 10)
scores = np.zeros(C_penalty.shape)
std = np.zeros(C_penalty.shape)

for i, C in enumerate(C_penalty):
    linear_clf = SVC(C=C, kernel='linear')
    score      = cross_val_score(linear_clf, X, y, cv=5)
    scores[i]  = score.mean()
    std[i]     = score.std()
    print(f'{i+1}/{C_penalty.size} finished')

svm_linear = np.vstack((C_penalty, scores, std)).T
np.save('svm_linear', svm_linear)
