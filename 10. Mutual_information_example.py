from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

cv = CountVectorizer()
X_train = cv.fit_transform(train.data)
X_test = cv.transform(test.data)
Y_train = train.target
Y_test = test.target

mi = mutual_info_classif(X_train, Y_train)
limit = sorted(mi)[-100]
for word, mi_value in zip(cv.get_feature_names(), mi):
  if mi_value >= limit:
    print(f"{word}\t{mi_value:^2.3}")