from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from xgboost import XGBRegressor



X_train, X_test, Y_train, Y_test = train_test_split(*load_boston(return_X_y=True), test_size=0.33, random_state=42)

print(load_boston().DESCR)

regressors = [
    KNeighborsRegressor(3), # метод ближайшего соседа
    LinearSVR(), # метод опорных векторов
    GaussianProcessRegressor(1.0 * RBF(1.0)), # гауссовский классификатор?
    DecisionTreeRegressor(), # решающие деревья
    RandomForestRegressor(), # метод случайных лесов
    MLPRegressor(), # нейронные сети (ОНИ САМЫЕ!)
    AdaBoostRegressor(), # адабуст (адаптивный бустинг)
    XGBRegressor()
    ]
for regressor in regressors:
    regressor.fit(X_train, Y_train)
    predicted_train = regressor.predict(X_train)
    predicted_test = regressor.predict(X_test)
    mae = mean_absolute_error(Y_test, predicted_test)
    print(f"{regressor.__class__.__name__}\t{mae}")