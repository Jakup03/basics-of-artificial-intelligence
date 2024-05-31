import idx2numpy
import numpy as np


# @staticmethod
# def deep_neural_network(inputs, hidden_weights, output_weights):
#     hidden = NeuralNetwork.neural_network_relu(inputs, hidden_weights)
#     outputs = NeuralNetwork.neural_network(hidden, output_weights)
#     return outputs
#
#
# @staticmethod
# def calculations(
#         inputs, hidden_weights, output_weights, goal_outputs, epochs, alpha
# ):
#     for epoch in range(epochs):
#         for series in range(len(inputs)):
#             hidden_layer = NeuralNetwork.neural_network_relu(
#                 inputs[series].reshape(-1, 1), hidden_weights
#             )
#
#             outputs = NeuralNetwork.neural_network(hidden_layer, output_weights)
#             outputs_delta = (
#                     2 / len(outputs) * (outputs - goal_outputs[series].reshape(-1, 1))
#             )
#
#             hidden_delta = np.matmul(output_weights.T, outputs_delta)
#             hidden_delta = hidden_delta * NeuralNetwork.derivative_relu(
#                 hidden_layer
#             )
#
#             output_weights_delta = np.matmul(outputs_delta, hidden_layer.T)
#             hidden_weights_delta = np.matmul(
#                 hidden_delta, inputs[series].reshape(-1, 1).T
#             )
#
#             output_weights = output_weights - alpha * output_weights_delta
#             hidden_weights = hidden_weights - alpha * hidden_weights_delta
#
#             print(f"i {epoch + 1} series {series + 1}:\n{outputs}")
#         print()

class ActivationFunctions:
    @staticmethod
    def neural_network_relu(inputs, weights):
        layer = np.matmul(weights, inputs)
        return np.maximum(0, layer)

    @staticmethod
    def neural_network_sigmoid(inputs, weights):
        layer = np.matmul(weights, inputs)
        return 1 / (1 + np.exp(-layer))

    @staticmethod
    def neural_network_tanh(inputs, weights):
        layer = np.matmul(weights, inputs)
        return np.tanh(layer)

    @staticmethod
    def neural_network_softmax(inputs, weights):
        layer = np.matmul(weights, inputs)
        e_x = np.exp(layer - np.max(layer))
        return e_x / e_x.sum()

    @staticmethod
    def neural_network_softmax_matrix(inputs, weights):
        layer = np.matmul(weights, inputs)
        e_x = np.exp(layer - np.max(layer, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def neural_network(inputs, weights):
        return np.matmul(weights, inputs)

    @staticmethod
    def neural_network_activation(inputs, weights, function):
        match function:
            case "relu":
                return ActivationFunctions.neural_network_relu(inputs, weights)
            case "sigmoid":
                return ActivationFunctions.neural_network_sigmoid(inputs, weights)
            case "tanh":
                return ActivationFunctions.neural_network_tanh(inputs, weights)
            case "softmax":
                return ActivationFunctions.neural_network_softmax(inputs, weights)
            case _:
                return ActivationFunctions.neural_network(inputs, weights)

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
    def derivative_tanh(layer):
        return 1 - (layer ** 2)

    @staticmethod
    def derivative_function(layer, function):
        match function:
            case "relu":
                return ActivationFunctions.derivative_relu(layer)
            case "sigmoid":
                return ActivationFunctions.derivative_sigmoid(layer)
            case "tanh":
                return ActivationFunctions.derivative_tanh(layer)
            case _:
                return np.ones((len(layer), 1))


class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.input_weights = np.random.uniform(-0.1, 0.1, (num_outputs, num_inputs))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.bin_layers = []
        self.functions = []
        self.weights = []

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
    def prepare_colors(data):
        inputs = np.array(data[:, :3])
        outputs = np.array(data[:, 3])
        goal_outputs = np.zeros((len(outputs), 4), dtype=np.float32)

        goal_outputs[outputs == 1] = (1.0, 0.0, 0.0, 0.0)
        goal_outputs[outputs == 2] = (0.0, 1.0, 0.0, 0.0)
        goal_outputs[outputs == 3] = (0.0, 0.0, 1.0, 0.0)
        goal_outputs[outputs == 4] = (0.0, 0.0, 0.0, 1.0)

        return inputs, goal_outputs

    def add_layer(self, n, weight_min_value=-0.1, weight_max_value=0.1, activation_function="None"):
        if len(self.layers) == 0:
            self.input_weights = np.random.uniform(-0.1, 0.1, (n, self.num_inputs))
        else:
            prev_layer = self.layers[-1]
            self.weights[-1] = np.random.uniform(weight_min_value, weight_max_value, (n, len(prev_layer)))

        hidden = np.zeros((n, 1))
        hidden_weights = np.random.uniform(weight_min_value, weight_max_value, (self.num_outputs, n))
        self.layers.append(hidden)
        self.functions.append(activation_function)
        self.weights.append(hidden_weights)

    def fit(self, inputs, goal_outputs, alpha=0.01, percent=0.5):
        for series in range(len(inputs)):
            prev_layer = inputs[series].reshape(-1, 1)
            prev_weights = self.input_weights

            for i in range(len(self.layers)):
                self.layers[i] = ActivationFunctions.neural_network_activation(prev_layer, prev_weights,
                                                                               self.functions[i])

                indices_to_zero = np.random.choice(len(self.layers[i]), int(len(self.layers[i]) * percent),
                                                   replace=False)
                self.layers[i][indices_to_zero] = 0
                self.layers[i][self.layers[i] != 0] *= 1 / percent
                bin_layer = (self.layers[i] != 0).astype(int)
                self.bin_layers.append(bin_layer)

                prev_layer = self.layers[i]
                prev_weights = self.weights[i]

            outputs = ActivationFunctions.neural_network_softmax(prev_layer, prev_weights)
            outputs_delta = 2 / len(outputs) * (outputs - goal_outputs[series].reshape(-1, 1))

            if len(self.weights) == 0:
                self.input_weights -= alpha * outputs_delta
                continue

            elif len(self.weights) == 1:
                hidden_delta = np.matmul(self.weights[0].T, outputs_delta)
                hidden_delta = hidden_delta * ActivationFunctions.derivative_function(self.layers[0], self.functions[0])
                hidden_delta *= self.bin_layers[0]

                output_weights_delta = np.matmul(outputs_delta, self.layers[0].T)
                input_weights_delta = np.matmul(hidden_delta, inputs[series].reshape(-1, 1).T)

                self.weights[0] = self.weights[0] - alpha * output_weights_delta
                self.input_weights = self.input_weights - alpha * input_weights_delta

                continue

            prev_delta = outputs_delta
            prev_layer = outputs
            for i in range(len(self.layers) - 1, 0, -1):
                hidden_delta = np.matmul(self.weights[i].T, prev_delta)
                hidden_delta = hidden_delta * ActivationFunctions.derivative_function(self.layers[i], self.functions[i])
                hidden_delta *= self.bin_layers[i]

                hidden_weights_delta = np.matmul(prev_delta, self.layers[i].T)
                self.weights[i] -= alpha * hidden_weights_delta

                prev_delta = hidden_delta
                prev_layer = self.layers[i]

            input_weights_delta = np.matmul(prev_delta, inputs[series].reshape(-1, 1).T)
            self.input_weights -= alpha * input_weights_delta

    def test_model(self, inputs, goal_outputs):
        hit = 0

        for series in range(len(inputs)):
            prev_layer = inputs[series]
            prev_weights = self.input_weights

            for i in range(len(self.layers)):
                self.layers[i] = ActivationFunctions.neural_network_activation(prev_layer, prev_weights,
                                                                               self.functions[i])
                prev_layer = self.layers[i]
                prev_weights = self.weights[i]

            outputs = ActivationFunctions.neural_network_softmax(prev_layer, prev_weights)

            if np.argmax(outputs) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(inputs)
        print(f"{np.round(avg * 100, 2)}%")

    @staticmethod
    def prepare_numbers(data):
        data = data.reshape(data.shape[0], 784)
        return data / 255.0

    @staticmethod
    def prepare_number_labels(data):
        return np.eye(10)[data]


class NeuralNetworkBatch:
    def __init__(self, num_inputs, num_outputs, batch_size):
        self.input_weights = np.random.uniform(-0.1, 0.1, (num_outputs, num_inputs))
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.layers = []
        self.bin_layers = []
        self.functions = []
        self.weights = []

    def add_layer(self, n, weight_min_value=-0.1, weight_max_value=0.1, activation_function="None"):
        if len(self.layers) == 0:
            self.input_weights = np.random.uniform(weight_min_value, weight_max_value, (n, self.num_inputs))
        else:
            prev_layer = self.layers[-1]
            self.weights[-1] = np.random.uniform(weight_min_value, weight_max_value, (n, len(prev_layer)))

        hidden = np.zeros((n, self.batch_size))
        hidden_weights = np.random.uniform(weight_min_value, weight_max_value, (self.num_outputs, n))
        self.layers.append(hidden)
        self.functions.append(activation_function)
        self.weights.append(hidden_weights)

    def fit(self, inputs, goal_outputs, alpha=0.01, percent=0.5):
        for batch in range(len(inputs)):
            prev_layer = inputs[batch].T
            prev_weights = self.input_weights

            for i in range(len(self.layers)):
                self.layers[i] = ActivationFunctions.neural_network_activation(prev_layer, prev_weights,
                                                                               self.functions[i])

                bin_layers = []
                for series in range(len(self.layers[i])):
                    indices_to_zero = np.random.choice(len(self.layers[i][series]),
                                                       int(len(self.layers[i][series]) * percent), replace=False)
                    self.layers[i][series][indices_to_zero] = 0
                    self.layers[i][series][self.layers[i][series] != 0] /= percent
                    bin_layer = (self.layers[i][series] != 0).astype(int)
                    bin_layers.append(bin_layer)
                self.bin_layers.append(bin_layers)

                prev_layer = self.layers[i]
                prev_weights = self.weights[i]

            outputs = ActivationFunctions.neural_network_softmax_matrix(prev_layer, prev_weights)
            outputs_delta = 2 / len(outputs) * (outputs - goal_outputs[batch].T) / self.batch_size

            if len(self.weights) == 0:
                self.input_weights -= alpha * outputs_delta
                continue

            elif len(self.weights) == 1:
                hidden_delta = np.matmul(self.weights[0].T, outputs_delta)
                hidden_delta = hidden_delta * ActivationFunctions.derivative_function(self.layers[0], self.functions[0])
                hidden_delta *= self.bin_layers[0]

                output_weights_delta = np.matmul(outputs_delta, self.layers[0].T)
                input_weights_delta = np.matmul(hidden_delta, inputs[batch])

                self.weights[0] = self.weights[0] - alpha * output_weights_delta
                self.input_weights = self.input_weights - alpha * input_weights_delta

                continue

            prev_delta = outputs_delta
            prev_layer = outputs
            for i in range(len(self.layers) - 1, 0, -1):
                hidden_delta = np.matmul(self.weights[i].T, prev_delta)
                hidden_delta = hidden_delta * ActivationFunctions.derivative_function(self.layers[i], self.functions[i])
                hidden_delta *= self.bin_layers[i]

                hidden_weights_delta = np.matmul(prev_delta, self.layers[i].T)
                self.weights[i] -= alpha * hidden_weights_delta

                prev_delta = hidden_delta
                prev_layer = self.layers[i]

            input_weights_delta = np.matmul(prev_delta, inputs[batch])
            self.input_weights -= alpha * input_weights_delta

    def test_model(self, inputs, goal_outputs):
        hit = 0

        for series in range(len(inputs)):
            prev_layer = inputs[series]
            prev_weights = self.input_weights

            for i in range(len(self.layers)):
                self.layers[i] = ActivationFunctions.neural_network_activation(prev_layer, prev_weights,
                                                                               self.functions[i])
                prev_layer = self.layers[i]
                prev_weights = self.weights[i]

            outputs = ActivationFunctions.neural_network_softmax_matrix(prev_layer, prev_weights)

            if np.argmax(outputs) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(inputs)
        print(f"{np.round(avg * 100, 2)}%")


def zad1():
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

    test_images = test_images[:10000]
    test_labels = test_labels[:10000]

    nn = NeuralNetwork(len(train_images[0]), 10)
    nn.add_layer(40, activation_function="relu")
    for _ in range(350):
        nn.fit(train_images[:1000], train_labels[:1000], alpha=0.005)
    nn.test_model(test_images, test_labels)

    nn2 = NeuralNetwork(len(train_images[0]), 10)
    nn2.add_layer(100, activation_function="relu")
    for _ in range(350):
        nn2.fit(train_images[:10000], train_labels[:10000], alpha=0.005)
    nn2.test_model(test_images, test_labels)

    nn3 = NeuralNetwork(len(train_images[0]), 10)
    nn3.add_layer(100, activation_function="relu")
    for _ in range(10):
        nn3.fit(train_images[:60000], train_labels[:60000], alpha=0.005)
    nn3.test_model(test_images, test_labels)


def zad2():
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

    test_images = test_images[:10000]
    test_labels = test_labels[:10000]

    batch_size = 100
    cut_train_images = np.array(
        [
            train_images[i: i + batch_size]
            for i in range(0, len(train_images), batch_size)
        ]
    )
    cut_train_labels = np.array(
        [
            train_labels[i: i + batch_size]
            for i in range(0, len(train_labels), batch_size)
        ]
    )

    nn = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn.add_layer(40, activation_function="relu")
    for _ in range(350):
        nn.fit(cut_train_images[:10], cut_train_labels[:10], alpha=0.1)
    nn.test_model(test_images, test_labels)

    nn2 = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn2.add_layer(100, activation_function="relu")
    for _ in range(350):
        nn2.fit(cut_train_images[:100], cut_train_labels[:100], alpha=0.1)
    nn2.test_model(test_images, test_labels)

    nn3 = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn3.add_layer(100, activation_function="relu")
    for _ in range(350):
        nn3.fit(cut_train_images[:600], cut_train_labels[:600], alpha=0.1)
    nn3.test_model(test_images, test_labels)


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

    test_images = test_images[:1000]
    test_labels = test_labels[:1000]

    batch_size = 100
    cut_train_images = np.array(
        [
            train_images[i: i + batch_size]
            for i in range(0, len(train_images), batch_size)
        ]
    )
    cut_train_labels = np.array(
        [
            train_labels[i: i + batch_size]
            for i in range(0, len(train_labels), batch_size)
        ]
    )

    nn = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn.add_layer(100, activation_function="relu")
    for i in range(350):
        nn.fit(cut_train_images[:10], cut_train_labels[:10], alpha=0.02)
    print("Relu: ", end="")
    nn.test_model(test_images, test_labels)

    nn = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn.add_layer(
        100, weight_min_value=-0.01, weight_max_value=0.01, activation_function="tanh"
    )
    for i in range(350):
        nn.fit(cut_train_images[:10], cut_train_labels[:10], alpha=0.02)
    print("Tanh: ", end="")
    nn.test_model(test_images, test_labels)

    nn = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn.add_layer(100, activation_function="sigmoid")
    for i in range(350):
        nn.fit(cut_train_images[:10], cut_train_labels[:10], alpha=0.02)
    print("Sigmoid: ", end="")
    nn.test_model(test_images, test_labels)

    nn = NeuralNetworkBatch(len(train_images[0]), 10, batch_size)
    nn.add_layer(100, activation_function="softmax")
    for i in range(350):
        nn.fit(cut_train_images[:10], cut_train_labels[:10], alpha=0.02)
    print("Softmax: ", end="")
    nn.test_model(test_images, test_labels)


task_functions = {
    "1": zad1,
    "2": zad2,
    "3": zad3,
}
while 1:
    wybor = input("Które zadanie: ")
    if wybor in task_functions:
        task_functions[wybor]()
        break
    else:
        print("Nieprawidłowy wybór zadania.")
