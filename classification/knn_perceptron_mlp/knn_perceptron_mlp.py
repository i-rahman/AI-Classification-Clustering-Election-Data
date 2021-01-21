import numpy as np
import random


class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB)**2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        self.X_train = X
        self.y_train = y

    def getNeighbors(self, X):
        allDistances = []
        for i in range(len(self.X_train)):
            dist = self.distance(X, self.X_train[i])
            allDistances.append((self.X_train[i], self.y_train[i], dist))
        allDistances.sort(key=lambda x: x[2])
        neighbors = []
        for i in range(self.k):
            neighbors.append((allDistances[i][0], allDistances[i][1]))
        return neighbors

    def getModeKlabels(self, neighbors):
        labelCount = {}
        for x in range(len(neighbors)):
            label = neighbors[x][1]
            if label in labelCount:
                labelCount[label] += 1
            else:
                labelCount[label] = 1
        sortedlabelCount = sorted(labelCount.items(),
                                  key=lambda x: x[1], reverse=True)
        return sortedlabelCount[0][0]

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features

        predictions = []
        for x in range(len(X)):
            neighbors = self.getNeighbors(X[x])
            chosen_neighbor = self.getModeKlabels(neighbors)
            predictions.append(chosen_neighbor)
        return np.ravel(predictions)


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        dataSize = int(len(X))

        for i in range(steps):
            # picking random data point generated
            # better result comapred to dataPoint = i%dataSize
            dataPoint = random.choice(range(dataSize))
            predictionY = np.dot(X[dataPoint], self.w)+self.b
            if predictionY > 0:
                predictionY = 1
            else:
                predictionY = 0

            if predictionY != y[dataPoint]:
                self.w += X[dataPoint]*(y[dataPoint]-predictionY)*self.lr
                self.b += (y[dataPoint]-predictionY)*self.lr

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        result = []
        for data in X:
            res = np.dot(data, self.w)+self.b
            if res > 0:
                res = 1
            else:
                res = 0
            result.append(res)
        return np.array(result)


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if(i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # Write forward pass here
        self.input = input
        res = np.dot(input, self.w)+self.b
        return res

    def backward(self, gradients):
        # Write backward pass here
        deltaW = np.dot(self.input.T, gradients)
        gradientPrime = np.dot(gradients, self.w.T)
        self.w -= self.lr*deltaW
        self.b -= self.lr*gradients
        return gradientPrime


class Sigmoid:

    def __init__(self):
        self.x = None

    def forward(self, input):
        self.x = 1/(1+np.exp(-input))
        return self.x

    def backward(self, gradients):
        x = np.array(self.x*(1-self.x))
        return (gradients * x)
