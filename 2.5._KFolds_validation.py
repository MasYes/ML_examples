import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_digits
from sklearn import preprocessing, metrics
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

dataset = load_digits()

X = preprocessing.scale(dataset["data"])
y = dataset["target"]

n_folds = 10

folds = KFold(n_folds)

total_accuracy = 0

for train_inds, test_inds in folds.split(X):
    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    total_accuracy += metrics.precision_recall_fscore_support(y_test, svc.predict(X_test), average='macro')[0]

print("Average accuracy: " + str(total_accuracy/n_folds))
