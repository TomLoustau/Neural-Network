import pygame
from keras.datasets import mnist
import time
from NeuralNetwork import *

class Vizualizer():
    COUCHES_H_SIZE = [420, 520, 420]

    def __init__(self, w = 1280, h = 720):
        self.width = w
        self.height = h
        self.screen = None
        self.clock = pygame.time.Clock()
        self.running = True
        self.x_couches = [w//4, (w//2), 3 * (w // 4)]
        self.pos_neuron = [None , None, None]
        self.nb_neuron = [5, 8, 5]
        self.marges = [150, 150, 150]
        self.nl = NeuralNetwork()

    def screen_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill("black")
            self.draw_neural_network()
            self.neuron_activation()

            pygame.display.flip()

            self.clock.tick(60)
        pygame.quit()

    def draw_neural_network(self):
        #drawing of the neuron
        for i in range(len(self.pos_neuron)):
            marge = ((self.height - self.COUCHES_H_SIZE[i]) // 2)
            self.pos_neuron[i] = [(self.x_couches[i], y) for y in range(marge, self.COUCHES_H_SIZE[i] + marge, self.COUCHES_H_SIZE[i] // self.nb_neuron[i])]
            for pos in self.pos_neuron[i]:
                pygame.draw.circle(self.screen, "white", pos, 20)

        #drawing of the line
        for i in range(len(self.pos_neuron) - 1):
            for pos1 in self.pos_neuron[i]:
                for pos2 in self.pos_neuron[i+1]:
                    pygame.draw.line(self.screen, "white", pos1, pos2, 2)


    def neuron_activation(self):
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

        nl.predict(X_train)

        activations = [nl.A1[:self.nb_neuron[0], 0], nl.A2[:self.nb_neuron[1], 0], nl.A1[:self.nb_neuron[2], 0]]
        weights = [nl.W1, nl.W2, nl.W3]
        nb_couche = len(self.pos_neuron)

        """for i in range(nb_couche - 1):
            for j in range(len(self.pos_neuron[i])):
                for t in range(len(self.pos_neuron[i+1])):
                    pos_neuron_begin = self.pos_neuron[i][j]
                    pos_neuron_end =  self.pos_neuron[i+1][t]
                    pygame.draw.line(self.screen, (np.abs(activations[i+1][t]) * 255, np.abs(activations[i+1][t]) * 255, 0), pos_neuron_begin, pos_neuron_end, 3)
            """

        for i in range(nb_couche):
            for j, activation in enumerate(activations[i]):
                pygame.draw.circle(self.screen, (np.abs(activation) * 255, np.abs(activation) * 255, 0), self.pos_neuron[i][j], 18)
                if i < nb_couche - 1:
                    for t in range(len(self.pos_neuron[i + 1])):
                        pos_neuron_begin = self.pos_neuron[i][j]
                        pos_neuron_end = self.pos_neuron[i + 1][t]
                        pygame.draw.line(self.screen,
                                         ((np.abs(weights[i][j][t]) / np.max(weights[i])) * 255,  (np.abs(weights[i][j][t]) / np.max(weights[i])) * 255, 0),
                                         pos_neuron_begin, pos_neuron_end, 3)