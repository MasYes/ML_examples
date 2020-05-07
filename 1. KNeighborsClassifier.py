from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

digits = datasets.load_digits() # загрузили датасет

# рисуем данные
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

Y_train = digits.target[:n_samples//2]
Y_test = digits.target[n_samples//2:]

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=2) # объявляем метод
knn.fit(X_train, Y_train) # обучаем метод
predicted_train = knn.predict(X_train) # получаем предсказания на тренировочном множестве
predicted_test = knn.predict(X_test) # получаем предсказания на тестовом множестве
print("precision, recall, f1 score")
print("Training set:\n", metrics.precision_recall_fscore_support(Y_train, predicted_train, average='macro')) # усредняем по классам
print("Testing set:\n", metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro'))


print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted_test))

errors = 0
for image, label, prediction, in zip(digits.images[n_samples//2:], digits.target[n_samples//2:], predicted_test):
    if label == prediction:
      continue
    plt.subplot(2, 4, errors + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
    errors += 1
    if errors == 4:
      break

plt.show()
