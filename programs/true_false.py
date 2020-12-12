import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import xgboost as xgb

np.random.seed(40)

def true_false_map(predictions, y_test, method):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and y_test[i] ==1:
            true_positive += 1
        elif predictions[i] == 1 and y_test[i] == 0:
            false_positive += 1
        elif predictions[i] == 0 and y_test[i] == 0:
            true_negative += 1
        elif predictions[i] == 0 and y_test[i] == 1:
            false_negative += 1

    import pandas as pd
    results = np.array([[true_negative, true_positive], [false_negative, false_positive]])
    results = pd.DataFrame(results, index=['True', 'False'], columns=['Negative', 'Positive'])
    print(results)
    sns.heatmap(results, annot=True)
    plt.title(method)
    plt.show()


    print('Number of positive cancer cases:', len(np.where(y_test == 1)[0]))
    print('Number of negative cancer cases:', len(np.where(y_test == 0)[0]))
    print('Number of wrong predictions:', false_negative+false_positive)


# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
print(X_train.shape)
print(X_test.shape)

# Logistic Regression
logreg = LogisticRegression(solver='sag', max_iter=5000)
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression: {:.3f}".format(logreg.score(X_test,y_test)))

predictions = logreg.predict(X_test)


true_false_map(predictions, y_test, 'Logistic Regression')


# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(X_train, y_train)
print("Test set accuracy with Decision Trees: {:.3f}".format(deep_tree_clf.score(X_test,y_test)))

predict_tree = deep_tree_clf.predict(X_test)

true_false_map(predict_tree, y_test, 'Decision Tree')

#xgboost

xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train,y_train)
print("Test set accuracy with XGBoost: {:.3f}".format(xg_clf.score(X_test,y_test)))

predict_xgb = xg_clf.predict(X_test)

true_false_map(predict_xgb, y_test, 'XG-boost')

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
predict_svm = svm_clf.predict(X_test)
print("Test set accuracy with SVM: {:.3f}".format(svm_clf.score(X_test,y_test)))

true_false_map(predict_svm, y_test, 'Support Vector Machine')



#now we use scaled data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
logreg = LogisticRegression(solver='sag', max_iter=500)
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy with Logistic Regression scaled: {:.3f}".format(logreg.score(X_test_scaled,y_test)))

predictions = logreg.predict(X_test_scaled)


#true_false_map(predictions, y_test, 'Logistic Regression scaled')


# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(X_train_scaled, y_train)
print("Test set accuracy with Decision Trees scaled: {:.3f}".format(deep_tree_clf.score(X_test_scaled,y_test)))

predict_tree = deep_tree_clf.predict(X_test_scaled)

#true_false_map(predict_tree, y_test, 'Decision Tree scaled')

#xgboost

xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train_scaled,y_train)
print("Test set accuracy with XGBoost scaled: {:.3f}".format(xg_clf.score(X_test_scaled,y_test)))

predict_xgb = xg_clf.predict(X_test_scaled)

#true_false_map(predict_xgb, y_test, 'XG-boost scaled')

svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
print("Test set accuracy with SVM scaled: {:.3f}".format(svm_clf.score(X_test_scaled,y_test)))

predict_svm = svm_clf.predict(X_test_scaled)

#true_false_map(predict_svm, y_test, 'Support Vector Machine scaled')