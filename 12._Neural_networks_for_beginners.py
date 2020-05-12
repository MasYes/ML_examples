from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import metrics

digits = datasets.load_digits()

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]

classifier = MLPClassifier(hidden_layer_sizes=(250,200,150), learning_rate_init=0.001, max_iter=100000)
classifier.fit(X_train, Y_train)
predicted_train = classifier.predict(X_train)
predicted_test = classifier.predict(X_test)
accuracy = metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro')[0]
print(f"{classifier.__class__.__name__}\t{accuracy}")