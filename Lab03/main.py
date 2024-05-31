import idx2numpy
import numpy as np

import numpy as np

# Zad 1
# inputs = [[2]]
# weights = [[0.5]]
# goal_output = [[0.8]]
# alpha = 0.1

# NeuralNetwork.learn(inputs, weights, alpha, goal_output, 5)
# print()
# NeuralNetwork.learn(inputs, weights, alpha, goal_output, 20)
# print()

# Zad 2
# inputs = np.array([[0.5, 0.75, 0.1], [0.1, 0.3, 0.7], [0.2, 0.1, 0.6], [0.8, 0.9, 0.2]])
# weights = np.array(
#     [[0.1, 0.1, -0.3], [0.1, 0.2, 0], [0, 0.7, 0.1], [0.2, 0.4, 0], [-0.3, 0.5, 0.1]]
# )
# goal_output = np.array(
#     [
#         [0.1, 1, 0.1, 0, -0.1],
#         [0.5, 0.2, -0.5, 0.3, 0.7],
#         [0.1, 0.3, 0.2, 0.9, 0.1],
#         [0.7, 0.6, 0.2, -0.1, 0.8],
#     ]
# )
# alpha = 0.01

# NeuralNetwork.learn(inputs, weights, alpha, goal_output, 1000)

# Zad 3
class NeuralNetwork:
    @staticmethod
    def neural_network(inputs, weights):
        return np.matmul(weights, inputs)

    @staticmethod
    def load_data(filename):
        return np.loadtxt(filename)

    @staticmethod
    def learn(inputs, input_weights, alpha, goal_output, epochs):
        updatedWeights = input_weights
        total_errors = np.zeros((len(inputs),), dtype=np.double)
        for epoch in range(epochs):
            for series in range(len(inputs)):
                prediction = NeuralNetwork.neural_network(inputs[series], updatedWeights)
                delta = 2 * np.outer(
                    ((prediction - goal_output[series]) * 1 / len(goal_output[series])),
                    inputs[series],
                )
                updatedWeights -= delta * alpha

                total_errors[series] = np.sum((prediction - goal_output[series]) ** 2) / len(
                    prediction
                )
            print(f"{epoch + 1}: {np.sum(total_errors)}")

    @staticmethod
    def train_color(inputs, input_weights, alpha, goal_output, epochs):
        updatedWeights = input_weights
        total_errors = np.zeros((len(inputs),), dtype=np.double)
        for epoch in range(epochs):
            for series in range(len(inputs)):
                prediction = NeuralNetwork.neural_network(inputs[series], updatedWeights)
                delta = 2 * np.outer(
                    ((prediction - goal_output[series]) * 1 / len(goal_output[series])),
                    inputs[series],
                )
                updatedWeights -= delta * alpha

                total_errors[series] = np.sum((prediction - goal_output[series]) ** 2) / len(
                    prediction
                )
                # print(np.argmax(prediction))
        return updatedWeights

    @staticmethod
    def test_color(inputs, input_weights, goal_outputs):
        correct_predictions = 0

        for series in range(len(inputs)):
            prediction = NeuralNetwork.neural_network(inputs[series], input_weights)
            print(prediction)
            if np.argmax(prediction) == np.argmax(goal_outputs[series]):
                correct_predictions += 1

        accuracy = correct_predictions / len(inputs)
        # print(accuracy)
        print(f"{np.round(accuracy * 100, 2)}%")

    @staticmethod
    def prepare_data(data):
        inputs = np.array(data[:, :3])
        outputs = np.array(data[:, 3])
        goal_outputs = np.zeros((len(outputs), 4), dtype=np.float32)

        goal_outputs[outputs == 1] = (1.0, 0.0, 0.0, 0.0)
        goal_outputs[outputs == 2] = (0.0, 1.0, 0.0, 0.0)
        goal_outputs[outputs == 3] = (0.0, 0.0, 1.0, 0.0)
        goal_outputs[outputs == 4] = (0.0, 0.0, 0.0, 1.0)

        return inputs, goal_outputs


# train_data = NeuralNetwork.load_data("colors_train.txt")
# test_data = NeuralNetwork.load_data("colors_test.txt")
#
# train_inputs, train_outputs = NeuralNetwork.prepare_data(train_data)
# test_inputs, test_outputs = NeuralNetwork.prepare_data(test_data)
#
# weights = np.random.uniform(low=0, high=1.0, size=(4, 3))
#
# trained_weights = NeuralNetwork.train_color(
#     train_inputs, weights, 0.01, train_outputs, 100
# )
# NeuralNetwork.test_color(test_inputs, trained_weights, test_outputs)

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.input_weights = np.random.uniform(-0.1, 0.1, (num_outputs, num_inputs))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.weights = []

    @staticmethod
    def neural_network_relu(inputs, weights):
        layer = np.matmul(weights, inputs)
        return np.maximum(0, layer)

    @staticmethod
    def neural_network_sigmoid(inputs, weights):
        layer = np.matmul(weights, inputs)
        return 1 / (1 + np.exp(-layer))

    @staticmethod
    def neural_network(inputs, weights):
        return np.matmul(weights, inputs)

    @staticmethod
    def neural_network_activation(inputs, weights, function):
        match (function):
            case "relu":
                return NeuralNetwork.neural_network_relu(inputs, weights)
            case "sigmoid":
                return NeuralNetwork.neural_network_sigmoid(inputs, weights)
            case _:
                return NeuralNetwork.neural_network(inputs, weights)

    @staticmethod
    def derivative_relu(layer):
        output = np.copy(layer)
        output[layer > 0] = 1
        output[layer <= 0] = 0
        return output

    @staticmethod
    def derivative_sigmoid(layer):
        return layer * (1 - layer)

    @staticmethod
    def derivative_function(layer, function):
        match (function):
            case "relu":
                return NeuralNetwork.derivative_relu(layer)
            case "sigmoid":
                return NeuralNetwork.derivative_sigmoid(layer)
            case _:
                return layer

    @staticmethod
    def deep_neural_network(inputs, hidden_weights, output_weights):
        hidden = NeuralNetwork.neural_network_relu(inputs, hidden_weights)
        outputs = NeuralNetwork.neural_network(hidden, output_weights)
        return outputs

    @staticmethod
    def calculations(
        inputs, hidden_weights, output_weights, goal_outputs, epochs, alpha
    ):
        for epoch in range(epochs):
            for series in range(len(inputs)):
                hidden_layer = NeuralNetwork.neural_network_relu(
                    inputs[series].reshape(-1, 1), hidden_weights
                )

                outputs = NeuralNetwork.neural_network(hidden_layer, output_weights)
                outputs_delta = (
                    2 / len(outputs) * (outputs - goal_outputs[series].reshape(-1, 1))
                )

                hidden_delta = np.matmul(output_weights.T, outputs_delta)
                hidden_delta = hidden_delta * NeuralNetwork.derivative_relu(
                    hidden_layer
                )

                output_weights_delta = np.matmul(outputs_delta, hidden_layer.T)
                hidden_weights_delta = np.matmul(
                    hidden_delta, inputs[series].reshape(-1, 1).T
                )

                output_weights = output_weights - alpha * output_weights_delta
                hidden_weights = hidden_weights - alpha * hidden_weights_delta

                print(f"i {epoch + 1} series {series + 1}:\n{outputs}")
            print()

    def add_layer(
        self, n, weight_min_value=-0.1, weight_max_value=0.1, activation_function="None"
    ):
        if len(self.layers) == 0:
            self.input_weights = np.random.uniform(-0.1, 0.1, (n, self.num_inputs))

        else:
            prevLayer = self.layers[-1][0]
            self.weights[-1] = np.random.uniform(
                weight_min_value, weight_max_value, (n, len(prevLayer))
            )

        hidden = np.zeros((n, 1))
        hidden_weights = np.random.uniform(
            weight_min_value, weight_max_value, (self.num_outputs, n)
        )
        self.layers.append([hidden, activation_function])
        self.weights.append(hidden_weights)

    def fit(self, inputs, goal_outputs, alpha=0.01):
        for series in range(len(inputs)):
            prevLayer = inputs[series].reshape(-1, 1)
            prevWeights = self.input_weights

            for layer, weights in zip(self.layers, self.weights):
                layer[0] = NeuralNetwork.neural_network_activation(
                    prevLayer, prevWeights, layer[1]
                )
                prevLayer = layer[0]
                prevWeights = weights

            outputs = NeuralNetwork.neural_network(prevLayer, prevWeights)
            outputs_delta = (
                2 / len(outputs) * (outputs - goal_outputs[series].reshape(-1, 1))
            )

            if len(self.weights) == 0:
                self.input_weights -= alpha * outputs_delta
                continue

            elif len(self.weights) == 1:
                hidden_delta = np.matmul(self.weights[0].T, outputs_delta)
                hidden_delta = hidden_delta * NeuralNetwork.derivative_function(
                    self.layers[0][0], self.layers[0][1]
                )

                output_weights_delta = np.matmul(outputs_delta, self.layers[0][0].T)
                hidden_weights_delta = np.matmul(
                    hidden_delta, inputs[series].reshape(-1, 1).T
                )

                self.weights[0] = self.weights[0] - alpha * output_weights_delta
                self.input_weights = self.input_weights - alpha * hidden_weights_delta

                continue

            prevDelta = outputs_delta
            prevLayer = outputs
            for i in range(len(self.layers) - 1, 0, -1):
                nodes, function = self.layers[i]

                hidden_delta = np.matmul(self.weights[i].T, prevDelta)
                hidden_delta = hidden_delta * NeuralNetwork.derivative_function(
                    nodes, function
                )

                hidden_weights_delta = np.matmul(prevDelta, nodes.T)
                self.weights[i] -= alpha * hidden_weights_delta

                prevDelta = hidden_delta
                prevLayer = nodes

            input_weights_delta = np.matmul(prevDelta, inputs[series].reshape(-1, 1).T)
            self.input_weights -= alpha * input_weights_delta

    def save_weights(self, file_name):
        with open(file_name, "w") as file:
            file.write(f"{len(self.weights)}\n\n")

            file.write(f"{self.input_weights.shape[0]},{self.input_weights.shape[1]}\n")

            for row in self.input_weights:
                for col in row:
                    file.write(f"{col} ")
                file.write("\n")
            file.write("\n")

            for i, weight in enumerate(self.weights):
                file.write(
                    f"({weight.shape[0]},{weight.shape[1]}) {self.layers[i][1]} | "
                )
            file.write("\n\n")

            for weight in self.weights:
                for row in weight:
                    for col in row:
                        file.write(f"{col} ")
                    file.write("\n")
                file.write("\n")

    def load_weights(self, file_name):
        with open(file_name, "r") as file:
            length = int(file.readline())
            file.readline()

            input_line = file.readline()
            self.num_inputs = int(input_line.split(",")[1])
            self.num_outputs = int(input_line.split(",")[0])

            self.input_weights = []
            for row in range(self.num_outputs):
                line = file.readline()
                self.input_weights.append([np.double(value) for value in line.split()])
            self.input_weights = np.array(self.input_weights)
            file.readline()

            datas = file.readline().split("|")[:-1]
            size_list = []
            functions = []
            for data in datas:
                size, function = data.split()
                rows, cols = size.split(",")
                rows = int(list(rows)[1])
                cols = int(list(cols)[0])
                size_list.append((rows, cols))
                functions.append(function)

            self.weights = []
            self.layers = []
            for i in range(length):
                file.readline()
                weight_rows = []
                for row in range(size_list[i][0]):
                    weight_rows.append(
                        [np.double(string) for string in file.readline().split()]
                    )
                self.weights.append(weight_rows)
                self.layers.append([np.zeros((len(weight_rows[0]), 1)), functions[i]])

    @staticmethod
    def prepare_numbers(data):
        data = data.reshape(data.shape[0], 784)
        return data / 255.0

    @staticmethod
    def prepare_number_labels(data):
        return np.eye(10)[data]

    @staticmethod
    def prepare_colors(data):
        inputs = np.array(data[:, :3])
        outputs = np.array(data[:, 3])
        goal_outputs = np.zeros((len(outputs), 4), dtype=np.float32)

        goal_outputs[outputs == 1] = (1.0, 0.0, 0.0, 0.0)
        goal_outputs[outputs == 2] = (0.0, 1.0, 0.0, 0.0)
        goal_outputs[outputs == 3] = (0.0, 0.0, 1.0, 0.0)
        goal_outputs[outputs == 4] = (0.0, 0.0, 0.0, 1.0)

        return inputs, goal_outputs

    def test_model(self, inputs, goal_outputs):
        hit = 0

        for series in range(len(inputs)):
            prevLayer = inputs[series]
            prevWeights = self.input_weights

            for layer, weights in zip(self.layers, self.weights):
                layer[0] = NeuralNetwork.neural_network_activation(
                    prevLayer, prevWeights, layer[1]
                )
                prevLayer = layer[0]
                prevWeights = weights

            outputs = NeuralNetwork.neural_network(prevLayer, prevWeights)

            if np.argmax(outputs) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(inputs)
        print(f"{np.round(avg * 100, 2)}%")

    @staticmethod
    def load_data(filename):
        return np.loadtxt(filename)


def zad1():
    inputs = np.array(
        [[0.5, 0.75, 0.1], [0.1, 0.3, 0.7], [0.2, 0.1, 0.6], [0.8, 0.9, 0.2]]
    )
    hidden_weights = np.array(
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1],
        ]
    )
    output_weights = np.array(
        [
            [0.7, 0.9, -0.4, 0.8, 0.1],
            [0.8, 0.5, 0.3, 0.1, 0.0],
            [-0.3, 0.9, 0.3, 0.1, -0.2],
        ]
    )

    for data in inputs:
        print(NeuralNetwork.deep_neural_network(data, hidden_weights, output_weights))


def zad2():
    inputs = np.array(
        [[0.5, 0.75, 0.1], [0.1, 0.3, 0.7], [0.2, 0.1, 0.6], [0.8, 0.9, 0.2]]
    )
    goal_outputs = np.array(
        [[0.1, 1.0, 0.1], [0.5, 0.2, -0.5], [0.1, 0.3, 0.2], [0.7, 0.6, 0.2]]
    )
    hidden_weights = np.array(
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1],
        ]
    )
    output_weights = np.array(
        [
            [0.7, 0.9, -0.4, 0.8, 0.1],
            [0.8, 0.5, 0.3, 0.1, 0.0],
            [-0.3, 0.9, 0.3, 0.1, -0.2],
        ]
    )
    alpha = 0.01

    NeuralNetwork.calculations(
        inputs, hidden_weights, output_weights, goal_outputs, 50, alpha
    )


def zad3():
    train_images_file = "train-images.idx3-ubyte"
    train_labels_file = "train-labels.idx1-ubyte"
    test_images_file = "t10k-images.idx3-ubyte"
    test_labels_file = "t10k-labels.idx1-ubyte"

    train_images = idx2numpy.convert_from_file(train_images_file)
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    test_images = idx2numpy.convert_from_file(test_images_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)

    train_labels = NeuralNetwork.prepare_number_labels(train_labels)
    test_labels = NeuralNetwork.prepare_number_labels(test_labels)

    train_images = NeuralNetwork.prepare_numbers(train_images)
    test_images = NeuralNetwork.prepare_numbers(test_images)

    nn = NeuralNetwork(len(train_images[0]), 10)
    nn.add_layer(40, activation_function="relu")
    # nn.load_weights("weights_numbers.txt")
    for _ in range(10):
        nn.fit(train_images, train_labels)
    nn.test_model(test_images, test_labels)
    # nn.save_weights("weights_numbers.txt")


def zad4():
    loaded_train_data = NeuralNetwork.load_data("colors_train.txt")
    loaded_test_data = NeuralNetwork.load_data("colors_test.txt")

    train_inputs, train_outputs = NeuralNetwork.prepare_colors(loaded_train_data)
    test_inputs, test_outputs = NeuralNetwork.prepare_colors(loaded_test_data)

    nn = NeuralNetwork(len(train_inputs[0]), 4)
    nn.add_layer(10, activation_function="relu")
    nn.add_layer(10, activation_function="relu")
    # nn.load_weights("weights_color.txt")
    for _ in range(100):
        nn.fit(train_inputs, train_outputs, alpha=0.005)
    nn.test_model(test_inputs, test_outputs)
    # nn.save_weights("weights_color.txt")


task_functions = {
    "1": zad1,
    "2": zad2,
    "3": zad3,
    "4": zad4
}
while 1:
    wybor = input("Które zadanie: ")
    if wybor in task_functions:
        task_functions[wybor]()
        break
    else:
        print("Nieprawidłowy wybór zadania.")
