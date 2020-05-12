import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # another trick to hide messages

import numpy
from sklearn import datasets, preprocessing
from sklearn.metrics import zero_one_loss
from keras.models import Sequential
from keras.layers.core import Dense, Activation

digits = datasets.load_digits()

n_samples = len(digits.images)

X = preprocessing.scale(digits.images.reshape((n_samples, -1)))

X_train = X[:n_samples//2]
X_test = X[n_samples//2:]

y = numpy.eye(numpy.max(digits.target) + 1)[digits.target]

Y_train = y[:n_samples//2]
Y_test = y[n_samples//2:]

ann = Sequential()
ann.add(Dense(output_dim=256, input_dim=X.shape[1], init="glorot_uniform"))
ann.add(Activation("relu"))
ann.add(Dense(output_dim=10, init="glorot_uniform"))
ann.add(Activation("sigmoid"))
ann.compile(loss='categorical_crossentropy', optimizer='sgd')

ann.fit(X_train, Y_train, nb_epoch=50, batch_size=32, verbose=0)
# The predicted class is the output response with the largest value
y_pred = numpy.argmax(ann.predict(X_test), 1)
ann_error = zero_one_loss(digits.target[n_samples//2:], y_pred)

print(1 - ann_error)