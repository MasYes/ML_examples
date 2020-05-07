from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn import metrics

digits = datasets.load_digits() # загрузили датасет

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]


parameters = {'C':[1e-4, 1e-3, 0.1, 1], "tol" : [1e-10, 1e-6, 1e-5, 1e-4]}
grid = GridSearchCV(LinearSVC(max_iter=1000000), parameters, n_jobs=2)
grid.fit(X_train, Y_train)

print("Best parameters:", grid.best_params_)
predicted_train = grid.predict(X_train)
predicted_test = grid.predict(X_test)

print("LinearSVC with selected parameters:")
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))
