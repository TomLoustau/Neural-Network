import numpy as np
from keras.datasets import mnist

from NeuralNetwork import *
from manim import *

class ManimAnimation(Scene):
    def __init__(self):
        super().__init__()
        self.nl = NeuralNetwork(nb_neurons=(10,10))
        self.digit = None
        self.real_digit = None
        self.nb_neurons = [self.nl.W1.shape[0], self.nl.W2.shape[0], self.nl.W3.shape[0]]
        self.neurons = [[] for _ in range(len(self.nb_neurons))]
        self.train_neurons = [[] for _ in range(len(self.nb_neurons))]
        self.buff = self.camera.frame_height / (max(self.nb_neurons) * 4)
        self.radius = self.camera.frame_height / (max(self.nb_neurons) * 4)
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

        self.play(*[Create(n) for n in self.train_neurons[0]])
        self.play(LaggedStart(*[AnimationGroup(*[Create(line) for line in self.train_connexions[0]])], lag_ratio=0.025))
        self.play(*[Create(n) for n in self.train_neurons[1]])
        self.play(LaggedStart(*[AnimationGroup(*[Create(line) for line in self.train_connexions[1]])], lag_ratio=0.025))
        self.play(*[Create(n) for n in self.train_neurons[2]])
        self.wait()

    def draw_neuron(self):
        self.fill_neuron_list()
        self.set_place_first_neuron(self.neurons)
        self.set_place_neurons(self.neurons)

    def fill_neuron_list(self):
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i]):
                self.neurons[i].append(Circle(radius=self.radius))
                self.neurons[i][j].set_fill(YELLOW, opacity=1)
                self.neurons[i][j].set_stroke(color=YELLOW)

    def set_place_first_neuron(self, neurons):
        x = [-4, 0, 4]
        for i in range(len(neurons)):
            height =  (self.radius + self.buff) * self.nb_neurons[i]
            neurons[i][0].move_to(np.array([x[i], (height / 2) + self.nb_neurons[i] / 12, 1]))

    def set_place_neurons(self, neurons):
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i] - 1):
                neurons[i][j + 1].next_to(neurons[i][j], DOWN, buff=self.buff)

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

        self.digit = X_test[1]

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

        self.real_digit = y_test[1]

        y_train = self.nl.one_hot(y_train)

        self.nl.train(X_train, y_train, X_test, y_test, 24, 32, True)

    def forward_visualizing(self):

        test_digit = self.digit
        test_digit = test_digit / test_digit.max()
        test_digit = test_digit.reshape(-1, 784)
        test_digit = test_digit.T

        self.nl.predict(test_digit)
        weights = [self.nl.W2, self.nl.W3]
        activations = [self.nl.A1, self.nl.A2, self.nl.A3]

        #Actualisation de la couleur des poids
        for i in range(len(self.nb_neurons) - 1):
            for j in range(self.nb_neurons[i]):
                for t in range(self.nb_neurons[i + 1]):
                    weight = weights[i][t, j]
                    max_weight = weights[i].max()
                    normalize_weight = abs(weight) / max_weight
                    r = normalize_weight * 0.969
                    g = normalize_weight * 0.988
                    b = normalize_weight * 0.455
                    line = Line(self.neurons[i][j].get_center(), self.neurons[i+1][t].get_center(), color=ManimColor.from_rgb((r, g, b)))
                    line.set_stroke(width=3)
                    line.set_z_index(-1)
                    self.train_connexions[i].append(line)

        #Actualisation de la couleur des neurones
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i]):
                activation = activations[i][j][0]
                max_activation = activations[i].max()
                normalize_act = abs(activation) / max_activation
                r = normalize_act * 0.969
                g = normalize_act * 0.988
                b = normalize_act * 0.455
                self.train_neurons[i].append(Circle(radius=self.radius))
                self.train_neurons[i][j].set_fill(ManimColor.from_rgb((r, g, b)), opacity=1)
                self.train_neurons[i][j].set_stroke(color=ManimColor.from_rgb((r, g, b)))

        self.set_place_first_neuron(self.train_neurons)
        self.set_place_neurons(self.train_neurons)

        print(self.train_neurons[2][0].get_color())