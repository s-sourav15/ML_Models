import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, load_iris, make_regression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from collections import Counter
# import pandas as pd
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
    def __init__(self, lr = 0.001, n_iters = 1000, regularization = 'l1', lambdaP = 0.1, batchSize = 32):
        self.lr = lr
        self.regularization = regularization
        self.n_iters = n_iters
        self.lambdaP = lambdaP
        self.weights = None
        self.bias = None
        self.batch_size = batchSize

    def fit(self, X, y):
        samples, features = X.shape
        nBatches = samples // self.batch_size
        self.weights = np.zeros(features)
        self.bias = 0
        epochLosses = []
        for epoch in range(self.n_iters):

            for batch in range(nBatches):
                idx = np.random.choice(samples, size=self.batch_size, replace=True)
                batch_X = X[idx]
                batch_y = y[idx]
                z = np.dot(batch_X, self.weights) + self.bias
                preds = self._sigmoid(z)
                if self.regularization =='l1':
                    cost = (-1 /samples) * np.sum((batch_y) * np.log(preds) + (1 - batch_y) * np.log(1 - preds)) + self.lambdaP * self.weights
                    self.weights -= (self.lr / samples) * (np.dot((preds - batch_y), batch_X)) + self.lambdaP * np.sign(np.mean(self.weights))
                else:
                    cost = (-1 /samples) * np.sum((batch_y) * np.log(preds) + (1 - batch_y) * np.log(1 - preds)) + self.lambdaP * (self.weights ** 2)
                    self.weights -= (self.lr / samples) * (np.dot((preds - batch_y), batch_X)) + self.lambdaP * np.mean(self.weights) 
                self.bias -= (1 / samples) * np.sum(preds - batch_y)
                epochLosses.append(cost)
            if epoch % 100 == 0:
                print(f'loss for epoch {epoch} || {np.mean(epochLosses)}')

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.round(self._sigmoid(z)).astype(int)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# model = logisticRegression(regularization='l2')
# sc = StandardScaler()
# X, y = make_moons(n_samples=1000)
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8)
# xTrain = sc.fit_transform(xTrain)
# xTest = sc.transform(xTest)

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
    def __init__(self, k, n_iters = 1000, tol = 1e-4):
        self.k = k
        self.n_iters = n_iters
        self.tol = tol
    
    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace = False)]

        for i in range(self.n_iters):

            cluster_assignments = []
            for x in X:
                dist = np.linalg.norm(x - self.centroids, axis = 1)
                cluster_assignments.append(np.argmin(dist))
            
            for j in range(self.k):
                cluster_data_points = X[np.where(np.array(cluster_assignments) == j)]

                if len(cluster_data_points) > 0:
                    self.centroids[j] = np.mean(cluster_data_points, axis = 0)
                
            if i > 0 and np.equal(self.centroids, previousCentroids):
                break

            previousCentroids = np.copy(self.centroids)
        
    def predict(self, X):
        cluster_assignments = []
        for x in X:
            dist = np.linalg.norm(x - self.centorids, axis = 1)
            cluster_assignments.append(np.argmin(dist)
                                       )
                                       