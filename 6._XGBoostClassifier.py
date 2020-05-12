import warnings
warnings.filterwarnings("ignore") # игнорируем часть сообщений

from sklearn import datasets
from sklearn import metrics
from xgboost import XGBClassifier

digits = datasets.load_digits() # загрузили датасет

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]

xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
predicted_train = xgb.predict(X_train)
predicted_test = xgb.predict(X_test)
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))
