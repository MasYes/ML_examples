from sklearn.cluster import KMeans
from sklearn import metrics, datasets

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

X = data
Y = digits.target

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)
predictions = kmeans.predict(X)
print("First 10 elements:                 " + ", ".join([str(x) for x in Y[:10]]))
print("Clusters of the first 10 elements: " + ", ".join([str(x) for x in predictions[:10]]))
print("confusion matrix:")
print(metrics.confusion_matrix(Y, predictions))
