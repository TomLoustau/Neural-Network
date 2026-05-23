import numpy as np
from tensorflow.keras.datasets import mnist
from NeuralNetwork import *
import matplotlib.pyplot as plt

def main():
    print("Chagement de mnist...")
    nl = NeuralNetwork(0.05)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("mnist chargée")

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

    y_preds = []

    nl.train(X_train, y_train, X_test, y_test, 20, 256, True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(hspace=0.5)
    gs = fig.add_gridspec(3,5)
    for i in range(0,5):
        rand_int = np.random.random_integers(100)
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(X_test[:, rand_int].reshape(28,28), cmap="gray")
        ax.set_title(f"Prediction : {y_test[rand_int]}", color="yellow")
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Taux d'erreurs")
    ax.fill_between(range(len(nl.loss_history)), nl.loss_history, alpha=0.3, color='yellow')
    ax.grid(True, alpha=0.2)
    ax.plot(nl.loss_history, color="yellow")

    ax = fig.add_subplot(gs[2, :])
    ax.set_title("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Précision")
    ax.fill_between(range(len(nl.accuracy_history)), nl.accuracy_history, alpha=0.3, color='yellow')
    ax.grid(True, alpha=0.2)
    ax.plot(nl.accuracy_history, color="yellow")

    plt.show()
main()