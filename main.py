import numpy as np
from tensorflow.keras.datasets import mnist
from NeuralNetwork import *
from vizualizer import *
import matplotlib.pyplot as plt
from manimAnimation import *

def main():
    nl = NeuralNetwork(0.05)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[:10000]
    X_train = X_train / X_train.max()
    X_train = X_train.reshape(-1, 784)
    X_train = X_train.T

    X_test = X_test[:10000]
    X_test = X_test / X_test.max()
    X_test = X_test.reshape(-1, 784)
    X_test = X_test.T

    y_train = y_train[:10000]
    y_test = y_test[:10000]

    y_train = nl.one_hot(y_train)

    nl.train(X_train, y_train, X_test, y_test, 20, 256, True)
main()