import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('svm_sigmoid.npy')

gamma  = data[:, 0]
scores = data[:, 1]
std    = data[:, 2]

sns.set()
plt.plot(gamma, scores)
plt.fill_between(gamma, scores-std, scores+std, alpha=0.3)
#plt.xscale('log')
plt.xlabel(r'$a$')
plt.ylabel(r'Mean Accuracy')
plt.title(r'SVM w/ $K(\mathbf{x}_{i} ,\mathbf{x}_{j}) =\tanh( a\mathbf{x}_{i} \cdot \mathbf{x}_{j})$')
plt.show()
