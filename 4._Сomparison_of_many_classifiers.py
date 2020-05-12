import warnings
warnings.filterwarnings("ignore") # игнорируем часть сообщений

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

classifiers = [
    KNeighborsClassifier(3), # метод ближайшего соседа
    SVC(kernel="linear", C=0.025), # метод опорных векторов
    SVC(gamma=2, C=1),
    SVC(gamma=0.001),
    LinearSVC(),
    GaussianProcessClassifier(1.0 * RBF(1.0)), # гауссовский классификатор?
    DecisionTreeClassifier(max_depth=5), # решающие деревья
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), # метод случайных лесов
    MLPClassifier(alpha=1, max_iter=1000), # нейронные сети (ОНИ САМЫЕ!)
    AdaBoostClassifier(), # адабуст (адаптивный бустинг)
    GaussianNB(), # метод наивного байеса
    QuadraticDiscriminantAnalysis(), # дискриминантный анализ
    GridSearchCV(LinearSVC(max_iter=1000000), {'C':[1e-4, 1e-3, 0.1, 1], "tol" : [1e-10, 1e-6, 1e-5, 1e-4]}, n_jobs=2)
    ]
for classifier in classifiers:
    classifier.fit(X_train, Y_train)
    predicted_train = classifier.predict(X_train)
    predicted_test = classifier.predict(X_test)
    accuracy = metrics.precision_recall_fscore_support(Y_test, predicted_test, average='macro')[0]
    print(f"{classifier.__class__.__name__}\t{accuracy}")