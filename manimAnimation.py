import numpy as np
from keras.datasets import mnist

from NeuralNetwork import *
from manim import *

class ManimAnimation(Scene):
    def __init__(self):
        super().__init__()
        self.nl = NeuralNetwork()
        self.digit = None
        self.real_digit = None
        self.nb_neurons = [8, 8, 10]
        self.neurons = [[] for _ in range(len(self.nb_neurons))]
        self.buff = 0.3
        self.radius = 0.2
        self.connexions = [[] for _ in range(len(self.nb_neurons) - 1)]
        self.train_connexions = [[] for _ in range(len(self.nb_neurons) - 1)]
        self.texts = []
        self.train_model()

    def construct(self):
        self.draw_neuron()
        self.draw_weights()
        self.set_text()

        for layer in self.neurons:
            self.play(Create(VGroup(*layer)))
            self.bring_to_front(*layer)

        self.play(LaggedStart(*[Create(line) for layer in self.connexions for line in layer], lag_ratio=0.025))

        self.play(Create(VGroup(*self.texts)))

        self.forward_visualizing()

        self.play(LaggedStart(*[AnimationGroup(*[Create(line) for line in self.train_connexions[0]])], lag_ratio=0.025))
        self.play(LaggedStart(*[AnimationGroup(*[Create(line) for line in self.train_connexions[1]])], lag_ratio=0.025))

    def draw_neuron(self):
        self.fill_neuron_list()
        self.set_place_first_neuron()
        self.set_place_neurons()

    def fill_neuron_list(self):
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i]):
                self.neurons[i].append(Circle(radius=self.radius))
                self.neurons[i][j].set_fill(YELLOW, opacity=1)
                self.neurons[i][j].set_stroke(color=ManimColor.from_rgb((179, 135, 16)))

    def set_place_first_neuron(self):
        x = [-4, 0, 4]
        for i in range(len(self.nb_neurons)):
            height =  (self.radius + self.buff) * self.nb_neurons[i]
            self.neurons[i][0].move_to(np.array([x[i], (height / 2) + self.nb_neurons[i] / 12, 1]))

    def set_place_neurons(self):
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i] - 1):
                self.neurons[i][j + 1].next_to(self.neurons[i][j], DOWN, buff=self.buff)

    def draw_weights(self):
        for i in range(len(self.nb_neurons) - 1):
            for j in range(self.nb_neurons[i]):
                for t in range(self.nb_neurons[i + 1]):
                    line = Line(self.neurons[i][j].get_center(), self.neurons[i+1][t].get_center(), color=YELLOW)
                    line.set_z_index(-1)
                    line.set_stroke(width=1.8)
                    self.connexions[i].append(line)

    def set_text(self):
        for i, neuron in enumerate(self.neurons[-1]):
            str_i = str(i)
            text = Text(str_i, font_size=25)
            self.texts.append(text)
            self.texts[i].next_to(neuron, RIGHT, buff=0.2)

    def train_model(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        self.digit = X_test[0]

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

        self.real_digit = y_test[0]

        y_train = self.nl.one_hot(y_train)

        self.nl.train(X_train, y_train, X_test, y_test, 20, 256, True)

    def forward_visualizing(self):

        test_digit = self.digit
        test_digit = test_digit / test_digit.max()
        test_digit = test_digit.reshape(-1, 784)
        test_digit = test_digit.T

        self.nl.predict(test_digit)
        weights = [self.nl.W2, self.nl.W3]
        #for j,layer in enumerate(self.connexions):
        #    for t,connexion in enumerate(layer):
        #        new_connexion = connexion.set_color(RED)
        #        self.train_connexions[j].append(new_connexion)
                #weights[i][j,t % self.nb_neurons[i+1]]

        for i in range(len(self.nb_neurons) - 1):
            for j in range(self.nb_neurons[i]):
                for t in range(self.nb_neurons[i + 1]):
                    weight = weights[i][j, t % self.nb_neurons[i + 1]]
                    max_weight = weights[i].max()
                    r = (weight / max_weight) * 247
                    g = (weight / max_weight) * 252
                    b = (weight / max_weight) * 116
                    line = Line(self.neurons[i][j].get_center(), self.neurons[i+1][t].get_center(), color=ManimColor.from_rgb((r, g, b)))
                    line.set_stroke(width=3)
                    line.set_z_index(-1)
                    self.train_connexions[i].append(line)
                    print(self.real_digit)