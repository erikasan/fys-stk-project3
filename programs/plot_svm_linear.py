import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('svm_linear.npy')

C_penalty = data[:, 0]
scores    = data[:, 1]
std       = data[:, 2]

sns.set()
plt.plot(C_penalty, scores, 'o-')
plt.fill_between(C_penalty, scores-std, scores+std, alpha=0.3)
plt.xscale('log')
plt.xlabel(r'$C$ - Regularization parameter')
plt.ylabel(r'Mean Accuracy')
plt.title(r'SVM w/ dot-product kernel')
plt.show()
