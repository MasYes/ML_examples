import warnings
warnings.filterwarnings("ignore") # игнорируем часть сообщений

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics

digits = datasets.load_digits() # загрузили датасет

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]

forests = [RandomForestClassifier(n_estimators=10),
            RandomForestClassifier(n_estimators=100),
            RandomForestClassifier(n_estimators=1000),
            RandomForestClassifier(n_estimators=10000)]

for classifier in forests:
    classifier.fit(X_train, Y_train)
    predicted_train = classifier.predict(X_train)
    predicted_test = classifier.predict(X_test)
    accuracy = metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro')[0]
    print(f"{classifier.__class__.__name__}\t{accuracy}")