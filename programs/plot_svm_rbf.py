import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('svm_rbf.npy')

gamma     = data[:, 0]
scores    = data[:, 1]
std       = data[:, 2]

sns.set()
plt.plot(gamma, scores)
plt.fill_between(gamma, scores-std, scores+std, alpha=0.3)
plt.xscale('log')
plt.xlabel(r'$1/2\sigma^2$')
plt.ylabel(r'Mean Accuracy')
plt.title(r'SVM w/ $K(\mathbf{x}_{i} ,\mathbf{x}_{j}) =\exp\left( -| \mathbf{x}_{i} -\mathbf{x}_{j}| ^{2} /2\sigma ^{2}\right)$')
plt.show()
