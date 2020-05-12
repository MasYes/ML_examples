import warnings
warnings.filterwarnings("ignore") # игнорируем часть сообщений

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor


X_train, X_test, Y_train, Y_test = train_test_split(*load_boston(return_X_y=True), test_size=0.33, random_state=42)

print("Original features")
print([float(x) for x in X_train[0]])

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) # а еще есть метод fit_transform!

X_test_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaled features")
print([f"{x:^1.3}" for x in X_train_scaled[0]])

print("\nResults on original data:")
svr = LinearSVR()
svr.fit(X_train, Y_train)
predicted_test = svr.predict(X_test)
mae = mean_absolute_error(Y_test, predicted_test)
print(f"{svr.__class__.__name__}\t{mae}")

xgb = RandomForestRegressor(random_state=42)
xgb.fit(X_train, Y_train)
predicted_test = xgb.predict(X_test)
mae = mean_absolute_error(Y_test, predicted_test)
print(f"{xgb.__class__.__name__}\t{mae}")

print("\nResults on scaled data:")


svr = LinearSVR()
svr.fit(X_train_scaled, Y_train)
predicted_test = svr.predict(X_test_scaled)
mae = mean_absolute_error(Y_test, predicted_test)
print(f"{svr.__class__.__name__}\t{mae}")

xgb = RandomForestRegressor(random_state=42)
xgb.fit(X_train_scaled, Y_train)
predicted_test = xgb.predict(X_test_scaled)
mae = mean_absolute_error(Y_test, predicted_test)
print(f"{xgb.__class__.__name__}\t{mae}")