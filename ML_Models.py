import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, load_iris, make_regression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

def featureScaling(X, scaling = 'standard'):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    minX = min(X)
    maxX = max(X)
    if scaling == 'standard':
        X = (X - mean) / std
    else:
        X = (X - minX) / (maxX - minX)
    return X


class LinearRegressionVector:
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def fit(self):
        x_t = np.transpose(self.x)
        x_inv = np.linalg.inv(np.matmul(x_t, self.x))
        x_inv = np.matmul(x_inv, x_t)
        theta = np.dot(x_inv, self.y)
        self.theta = np.round(theta, 2)

X = [[1, 1], [1, 2], [1, 3]]
y = [1, 2, 3]
# lr_vector = LinearRegressionVector(X, y)
# lr_vector.fit()
# print(lr_vector.theta)

class linearRegression:
    def __init__(self, n_iters, lr):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X ,y):
        samples, features = X.shape
        self.weights = np.zeros((features, 1))
        y = y.reshape(-1, 1)
        for epoch in range(1, self.n_iters):
            prediction = X @ self.weights
            error = prediction - y
            update = X.T @ error 
            self.weights -= self.lr * (update / samples)
        self.weights.flatten()
            # 
    def predict(self, X):
        predictions = X @ self.weights
        return np.round(predictions, 2)

# X, y = make_regression(n_samples=1000, n_features=4)
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)
# # xTrain = featureScaling(xTrain)
# sc = StandardScaler()
# xTrain = sc.fit_transform(xTrain)
# xTest = sc.transform(xTest)

# model = linearRegression(n_iters=10000, lr = 0.01)
# model.fit(xTrain, yTrain)
# preds = model.predict(xTest)
# print('r2 score:', r2_score(yTest, preds))


class logisticRegression:
    def __init__(self, lr=0.01, lambdaP = 0.001, iters = 1000, batch_size = 32, regularization = 'l1'):
        self.regul = regularization
        self.batch_size = batch_size
        self.iters = iters
        self.lr = lr
        self.lambdaP = lambdaP
    
    def fit(self, X, y):
        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0
        batches = samples // self.batch_size
        epoch_losses = []
        for epoch in range(self.iters):
            for _ in range(batches):
                idx = np.random.choice(samples, size=self.batch_size, replace=False)
                batch_X = X[idx]
                batch_y = y[idx]

                z = np.dot(batch_X, self.weights) + self.bias
                preds = self._sigmoid(z)
                if self.regul == 'l1':
                    cost = (-1 / samples) * np.sum(batch_y * np.log(preds) + (1- batch_y) *( np.log(1- preds))) + self.lambdaP * self.weights
                    self.weights -= (self.lr /samples) * (np.dot(preds - batch_y, batch_X)) + self.lambdaP * np.sign(np.mean(self.weights))
                else:
                    cost = (-1 / samples) * np.sum(batch_y * np.log(preds) + (1- batch_y)*( np.log(1- preds))) + self.lambdaP * (self.weights ** 2)
                    self.weights -= (self.lr / samples) * np.dot(preds - batch_y, batch_X) + self.lambdaP * np.mean(self.weights)
                self.bias -= (1 / samples) * np.sum(preds - batch_y)
                epoch_losses.append(cost)
            if epoch % 100 == 0:
                print(np.mean(epoch_losses))
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        preds = self._sigmoid(z)
        return np.round(preds).astype(int)
                        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

model = logisticRegression(regularization='l2')
sc = StandardScaler()
X, y = make_moons(n_samples=1000)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

# model.fit(xTrain, yTrain)
# preds = model.predict(xTest)
# print(accuracy_score(yTest, preds))


class KNN:
    def __init__(self, xTrain, yTrain, k, dist = 'euclidean'):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.dist = dist
        self.k = k
    def fit(self, xTest):
        preds = []
        for x in xTest:
            if self.dist == 'euclidean':
                distance = np.linalg.norm(self.xTrain - x, axis = 1)
            else:
                distance = np.sum(np.abs(self.xTrain - x, axis = 1))
            indices = np.argsort(distance)[:self.k]
            nearestLabels = self.yTrain[indices]
            label = Counter(nearestLabels).most_common(1)[0][0]
            preds.append(label)
        return preds
    

# sc = StandardScaler()
# X, y = load_iris()['data'], load_iris()['target']
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)
# xTrain = sc.fit_transform(xTrain)
# xTest = sc.transform(xTest)
# model = KNN(xTrain, yTrain=yTrain, k=10)
# preds = model.fit(xTest)
# print(accuracy_score(yTest, preds))

# df = pd.DataFrame({'true':yTest, 'predictions':preds})
# print(df[:20])

class KMeans:
    def __init__(self, k, iters =200, tol = 1e-4):
        self.k =k
        self.tol = tol
        self.iters = iters

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), size=self.k, replace=False)]

        for iter in range(self.iters):
            cluster_assignments = []

            for x in X:
                dist = np.linalg.norm(x - self.centroids, axis = 1)
                cluster_assignments.append(np.argmin(dist))

            for k in range(self.k):
                cluster_points = X[np.where(np.array(cluster_assignments) == k)]
                if len(cluster_points) > 0:
                    self.centroids[k] = np.mean(cluster_points, axis = 0)

            if iter > 0 and np.mean(self.centroids - previous) < self.tol:
                break
                
            previous = np.copy(self.centroids)


    def predict(self, X):
        cluster_assignments = []
        for x in X:
            dist = np.linalg.norm(x - self.centroids, axis = 1)
            cluster_assignments.append(np.argmin(dist))
        return cluster_assignments
    
kmeans = KMeans(k = 2)
x1 = np.random.randn(15, 2) + 5
x2 = np.random.randn(15, 2) - 5
X = np.concatenate([x1, x2], axis =0)
# kmeans.fit(X)
# print(kmeans.predict(X))


class bagging:
    def __init__(self, estimators, baseEstimator, replace =False):
        self.estimators = estimators
        self.baseEstimator = baseEstimator
        self.replace = replace
        self.models = []
    def fit(self, X, y):
        subsamples = self.subsample(X)
        for i in range(self.estimators):
            idx = subsamples[i]
            batch_X = X[idx]
            batch_y = y[idx]
            self.baseEstimator.fit(batch_X, batch_y)
            self.models.append(self.baseEstimator)
        
    def predict(self, X):
        predictions = []
        for est in self.models:
            preds = est.predict(X)
            predictions.append(preds)
        
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def subsample(self, X):
        m, n = X. shape
        size = m if self.replace else m // 2
        return [np.random.choice(len(X), size=size, replace=self.replace) for _ in range(self.estimators)]

    
sc = StandardScaler()
X, y = load_iris()['data'], load_iris()['target']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)
model = bagging(estimators=10, baseEstimator=DecisionTreeClassifier())
# model.fit(xTrain, yTrain)

# preds = model.predict(xTest)
# # preds = model.fit(xTest)
# print(accuracy_score(yTest, preds))


class PCA:
    def __init__(self, k):
        self.k = k
        self.eigenvalues= None
        self.eigenvectors = None
    
    def fit(self, X):
        X_centered = X - np.mean(X, axis = 0)

        self.covariance = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        self.eigenvalues = self.eigenvalues[:self.k]
        self.eigenvectors = self.eigenvectors[:, :self.k]
        totalVar = np.sum(eigenvalues)
        self.cumulative_variance = self.eigenvalues / totalVar
        return np.dot(X_centered, self.eigenvectors)

def verify_eigenvectors(
    cov_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that eigenvectors and eigenvalues are correct
    
    Av = λv where A is covariance matrix, v is eigenvector, λ is eigenvalue
    """
    for i in range(len(eigenvalues)):
        # Compute Av
        Av = np.dot(cov_matrix, eigenvectors[:, i])
        # Compute λv
        lv = eigenvalues[i] * eigenvectors[:, i]
        # Check if they're equal
        if not np.allclose(Av, lv, rtol=tolerance):
            return False
    return True

data = load_iris()
X = data['data']
y = data['target']
sc = StandardScaler()
X = sc.fit_transform(X)
print(X.shape)
pca = PCA(k = 2)
X_proj = pca.fit(X)
print(X_proj.shape)
# verify_eigenvectors(pca.covariance, pca.eigenValues, pca.eigenVectors)
        
print("\nExplained Variance Ratios:")
for i, ratio in enumerate(pca.cumulative_variance):
    print(f"Component {i+1}: {ratio:.4f}")
    
# Visualize transformation
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Data (First 2 Dimensions)')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(122)
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.5)
plt.title('PCA Transformed Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.show()