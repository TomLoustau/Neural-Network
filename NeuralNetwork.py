import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    nb_w: np.ndarray
    biais: np.ndarray

    def __init__(self, lr=0.05):
        self.lr = lr
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.Z1 = None
        self.Z2 = None
        self.Z3 = None
        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.vW1 = None
        self.vW2 = None
        self.vW3 = None
        self.vB1 = None
        self.vB2 = None
        self.vB3 = None
        self.loss_history = []
        self.accuracy_history = []
        self.init_weights()

    def init_weights(self):
        # Using He initialization for ReLU activation functions
        self.W1 = np.random.randn(128, 784) * np.sqrt(1 / 784)
        self.W2 = np.random.randn(64, 128) * np.sqrt(1 / 128)
        self.W3 = np.random.randn(10, 64) * np.sqrt(1 / 64)
        self.b1 = np.zeros((128, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((10, 1))

        # initialisation of all velocities
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vW3 = np.zeros_like(self.W3)
        self.vB1 = np.zeros_like(self.b1)
        self.vB2 = np.zeros_like(self.b2)
        self.vB3 = np.zeros_like(self.b3)

    def predict(self, X: np.ndarray):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.tanh(self.Z1)

        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.tanh(self.Z2)

        self.Z3 = self.W3 @ self.A2 + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def ReLU(self, y):
        return np.maximum(0, y)

    def tanh(self, y):
        return np.tanh(y)

    def softmax(self, y):
        epsilon = 1e-12
        exp = np.exp(y - np.max(y + epsilon, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def deriv_tanh(self, y):
        return 1 - np.tanh(y) ** 2

    def one_hot(self, y):
        one_hot = np.zeros((10, len(y)))
        one_hot[y, np.arange(len(y))] = 1
        return one_hot

    def decode_one_hot(self, y):
        decode = np.argmax(y, axis=0)
        return decode

    def loss(self, y_pred, y):
        epsilon = 1e-12
        return -np.mean(np.sum(y * np.log(y_pred + epsilon), axis=0))

    def back_propagation(self, X, y_oh):
        m = X.shape[1]  # m is the batch size here
        dZ3 = (1 / m) * (self.A3 - y_oh)
        dZ2 = self.W3.T @ dZ3 * self.deriv_tanh(self.Z2)
        dZ1 = self.W2.T @ dZ2 * self.deriv_tanh(self.Z1)

        grad_W3 = (dZ3 @ self.A2.T)
        self.vW3 = self.momentum(grad_W3, self.vW3)
        self.W3 = self.W3 - self.vW3 * self.lr

        grad_b3 = np.sum(dZ3, axis=1, keepdims=True)
        self.vB3 = self.momentum(grad_b3, self.vB3)
        self.b3 = self.b3 - self.vB3 * self.lr

        grad_W2 = (dZ2 @ self.A1.T)
        self.vW2 = self.momentum(grad_W2, self.vW2)
        self.W2 = self.W2 - self.vW2 * self.lr

        grad_b2 = np.sum(dZ2, axis=1, keepdims=True)
        self.vB2 = self.momentum(grad_b2, self.vB2)
        self.b2 = self.b2 - self.vB2 * self.lr

        grad_W1 = (dZ1 @ X.T)
        self.vW1 = self.momentum(grad_W1, self.vW1)
        self.W1 = self.W1 - self.vW1 * self.lr

        grad_b1 = np.sum(dZ1, axis=1, keepdims=True)
        self.vB1 = self.momentum(grad_b1, self.vB1)
        self.b1 = self.b1 - self.vB1 * self.lr

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, view_info):
        self.init_weights()
        m = X_train.shape[1]

        for epoch in range(epochs):
            idx = np.random.permutation(m)
            X_s, y_s = X_train[:, idx], y_train[:, idx]

            y_pred = self.predict(X_train)
            l = self.loss(y_pred, y_train)
            acc = self.accuracy(X_test, y_test)
            self.loss_history.append(l)
            for i in range(0, m, batch_size):
                self.accuracy_history.append(acc)
                X_batch = X_s[:, i:i + batch_size]
                y_batch = y_s[:, i:i + batch_size]
                y_pred = self.predict(X_batch)
                self.back_propagation(X_batch, y_batch)

            if view_info:
                print("Epoch : ", epoch, "/", epochs)
                print("Loss : ", l)
                print("Accuracy : ", acc)

    def accuracy(self, X_test, y_test):
        y_test_pred = self.predict(X_test)
        y_prediction = self.decode_one_hot(y_test_pred)
        correct = np.sum(y_test == y_prediction)
        return correct / y_test.shape[0]

    def plot_visualisation(self):
        plt.plot(self.loss_history)
        plt.plot(self.accuracy_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Apprentissage")
        plt.show()

    def momentum(self, grad, v):
        beta = 0.5
        v = beta * v + (1 - beta) * grad
        return v