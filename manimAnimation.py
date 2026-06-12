import numpy as np
from keras.datasets import mnist

from NeuralNetwork import *
from manim import *
from PIL import Image

class ManimAnimation(Scene):
    X_LAYER_1 = -3
    X_LAYER_2 = 1
    X_LAYER_3 = 5
    X_DIGIT = -6
    FONT_SIZE = 25
    DIGIT_PIXEL_SIZE = 0.08

    def __init__(self):
        super().__init__()
        self.img = None
        self.nl = NeuralNetwork(nb_neurons=(2,2))
        self.digit = None
        self.real_digit = None
        self.neuron_color = YELLOW_D
        self.nb_neurons = [self.nl.W1.shape[0], self.nl.W2.shape[0], self.nl.W3.shape[0]]
        self.neurons = [[] for _ in range(len(self.nb_neurons))]
        self.train_neurons = [[] for _ in range(len(self.nb_neurons))]
        self.buff = self.camera.frame_height / (max(self.nb_neurons) * 4)
        self.radius = self.camera.frame_height / (max(self.nb_neurons) * 4)
        self.connexions = [[] for _ in range(len(self.nb_neurons) - 1)]
        self.train_connexions = [[] for _ in range(len(self.nb_neurons) - 1)]
        self.texts = []

    def construct(self):
        self.train_model()

        grid = self.draw_digit()

        self.play(Create(grid))

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
        self.set_place_neurons(self.neurons)

    def fill_neuron_list(self):
        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i]):
                self.neurons[i].append(Circle(radius=self.radius))
                self.neurons[i][j].set_fill(self.neuron_color, opacity=1)
                self.neurons[i][j].set_stroke(color=YELLOW_B)

    def set_place_neurons(self, neurons):
        #Placing the first neuron place
        x = [self.X_LAYER_1, self.X_LAYER_2, self.X_LAYER_3]
        for i in range(len(neurons)):
            height = (self.radius + self.buff) * self.nb_neurons[i]
            neurons[i][0].move_to(np.array([x[i], (height / 1.5), 1]))

        for i in range(len(self.nb_neurons)):
            for j in range(self.nb_neurons[i] - 1):
                neurons[i][j + 1].next_to(neurons[i][j], DOWN, buff=self.buff)

    def draw_weights(self):
        for i in range(len(self.nb_neurons) - 1):
            for j in range(self.nb_neurons[i]):
                for t in range(self.nb_neurons[i + 1]):
                    line = Line(self.neurons[i][j].get_center(), self.neurons[i+1][t].get_center(), color=self.neuron_color)
                    line.set_z_index(-1)
                    line.set_stroke(width=1.8)
                    self.connexions[i].append(line)

    def set_text(self):
        for i, neuron in enumerate(self.neurons[-1]):
            str_i = str(i)
            text = Text(str_i, font_size=self.FONT_SIZE)
            self.texts.append(text)
            self.texts[i].next_to(neuron, RIGHT, buff=self.buff)

    def train_model(self):
        random_index = np.random.randint(1, 1000)
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        self.digit = X_test[random_index]

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

        self.real_digit = y_test[random_index]

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

        #Actualisation of the weights color
        for i in range(len(self.nb_neurons) - 1):
            max_weight = weights[i].max()
            for j in range(self.nb_neurons[i]):
                for t in range(self.nb_neurons[i + 1]):
                    weight = weights[i][t, j]
                    normalize_weight = abs(weight) / max_weight
                    r = normalize_weight * self.neuron_color[0]
                    g = normalize_weight * self.neuron_color[1]
                    b = normalize_weight * self.neuron_color[2]
                    line = Line(self.neurons[i][j].get_center(), self.neurons[i+1][t].get_center(), color=ManimColor.from_rgb((r, g, b)))
                    line.set_stroke(width=3)
                    line.set_z_index(-1)
                    self.train_connexions[i].append(line)

        #Actualisation of the neuron colors
        for i in range(len(self.nb_neurons)):
            max_activation = activations[i].max()
            for j in range(self.nb_neurons[i]):
                activation = activations[i][j][0]
                normalize_act = abs(activation) / max_activation
                r = normalize_act * self.neuron_color[0]
                g = normalize_act * self.neuron_color[1]
                b = normalize_act * self.neuron_color[2]
                self.train_neurons[i].append(Circle(radius=self.radius))
                self.train_neurons[i][j].set_fill(ManimColor.from_rgb((r, g, b)), opacity=1)
                self.train_neurons[i][j].set_stroke(color=ManimColor.from_rgb((r, g, b)))

        self.set_place_neurons(self.train_neurons)


    def draw_digit(self):
        middle_screen_y = (self.DIGIT_PIXEL_SIZE * len(self.digit)) / 2
        grid = VGroup()
        for i in range(len(self.digit)):
            for j in range(len(self.digit[0])):
                couleur = self.digit[i][j] / 255
                square = Square(self.DIGIT_PIXEL_SIZE)
                square.set_fill((ManimColor.from_rgb((couleur * self.neuron_color[0], couleur * self.neuron_color[1], couleur * self.neuron_color[2]))), opacity=1)
                square.set_stroke(width=0.5, color=ManimColor.from_rgb((couleur * self.neuron_color[0], couleur * self.neuron_color[1], couleur * self.neuron_color[2])), opacity=1)
                square.move_to((self.X_DIGIT + j * self.DIGIT_PIXEL_SIZE, middle_screen_y - i * self.DIGIT_PIXEL_SIZE, 0))
                grid.add(square)
        return grid