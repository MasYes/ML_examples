from sklearn.svm import LinearSVC, SVC
from sklearn import datasets
from sklearn import metrics

digits = datasets.load_digits() # загрузили датасет

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]

svc = LinearSVC()
svc.fit(X_train, Y_train)
predicted_train = svc.predict(X_train)
predicted_test = svc.predict(X_test)
print("Default LinearSVC")
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))

svc = LinearSVC(C=0.01)
svc.fit(X_train, Y_train)
predicted_train = svc.predict(X_train)
predicted_test = svc.predict(X_test)
print("\nLinearSVC, C=0.01")
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))

svc = SVC(gamma=0.001)
# SVC(kernel='linear') != LinearSVC !
svc.fit(X_train, Y_train)
predicted_train = svc.predict(X_train)
predicted_test = svc.predict(X_test)
print("\nSVC, gamma=0.001")
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))
