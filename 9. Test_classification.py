from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics


train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
print(train.data[0])
print(train.target_names)

cv = CountVectorizer()
X_train = cv.fit_transform(train.data)
X_test = cv.transform(test.data)
Y_train = train.target
Y_test = test.target

svc = LinearSVC()
svc.fit(X_train, Y_train)
predicted_train = svc.predict(X_train)
predicted_test = svc.predict(X_test)
print(metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro'))
print(metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))