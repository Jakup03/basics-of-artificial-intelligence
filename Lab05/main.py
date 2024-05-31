import time

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
    def relu(layer):
        return np.maximum(0, layer)

    @staticmethod
    def sigmoid(layer):
        return 1 / (1 + np.exp(-layer))

    @staticmethod
    def tanh(layer):
        return np.tanh(layer)

    @staticmethod
    def softmax(layer):
        e_x = np.exp(layer - np.max(layer))
        return e_x / e_x.sum()

    @staticmethod
    def softmax_matrix(layer):
        e_x = np.exp(layer - np.max(layer, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def neural_network_activation(layer, function):
        match (function):
            case "relu":
                return ActivationFunctions.relu(layer)
            case "sigmoid":
                return ActivationFunctions.sigmoid(layer)
            case "tanh":
                return ActivationFunctions.tanh(layer)
            case "softmax":
                return ActivationFunctions.softmax(layer)
            case _:
                return layer

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
        return 1 - (layer**2)

    @staticmethod
    def derivative_function(layer, function):
        match (function):
            case "relu":
                return ActivationFunctions.derivative_relu(layer)
            case "sigmoid":
                return ActivationFunctions.derivative_sigmoid(layer)
            case "tanh":
                return ActivationFunctions.derivative_tanh(layer)
            case _:
                return np.ones((len(layer), 1))


class NeuralNetwork:
    def __init__(self, output_weights):
        self.output_weights = output_weights

    @staticmethod
    def cut_image(image, mask_size, step=1):
        square_size = int(np.sqrt(mask_size))
        cut_size = square_size // 2

        start_x, end_x = cut_size, image.shape[1] - cut_size
        start_y, end_y = cut_size, image.shape[0] - cut_size
        cut_image = np.array(
            [
                image[
                    row - cut_size : row + cut_size + 1,
                    col - cut_size : col + cut_size + 1,
                ]
                for row in range(start_y, end_y, step)
                for col in range(start_x, end_x, step)
            ]
        )
        cut_image = cut_image.reshape(cut_image.shape[0], mask_size)

        return cut_image

    def convolution(self, input_image, mask, step=1, padding=0):
        image_height, image_width = input_image.shape
        mask = mask.reshape(-1, 1)
        mask_size = len(mask)

        new_image = np.zeros((image_height + 2 * padding, image_width + 2 * padding))
        new_image[
            padding : image_height + padding, padding : image_width + padding
        ] = input_image

        cut_new_image = NeuralNetwork.cut_image(new_image, mask_size, step)

        convolution_matrix = np.matmul(cut_new_image, mask)
        square_size = int(np.sqrt(convolution_matrix.shape[0]))
        convolution_matrix = convolution_matrix.reshape(square_size, square_size)

        return convolution_matrix

    def fit_przyklad(
        self,
        input_image,
        combined_kernel_weights,
        kernel_activation_function,
        goal_output,
        alpha=0.01,
    ):
        cut_image = NeuralNetwork.cut_image(
            input_image, len(combined_kernel_weights[0])
        )

        kernel_layer = np.matmul(cut_image, combined_kernel_weights.T)
        kernel_layer = ActivationFunctions.neural_network_activation(
            kernel_layer, kernel_activation_function
        )
        flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

        output_layer = np.matmul(self.output_weights, flatten_kernel_layer)
        output_delta = (
            2 / len(output_layer) * (output_layer - goal_output[:, np.newaxis])
        )

        kernel_delta = np.matmul(self.output_weights.T, output_delta)
        kernel_delta *= ActivationFunctions.derivative_function(
            kernel_delta, kernel_activation_function
        )
        kernel_delta_reshaped = kernel_delta.reshape(kernel_layer.shape)

        output_weights_delta = np.matmul(output_delta, flatten_kernel_layer.T)
        kernel_weights_delta = np.matmul(kernel_delta_reshaped.T, cut_image)

        self.output_weights = self.output_weights - alpha * output_weights_delta
        combined_kernel_weights = combined_kernel_weights - alpha * kernel_weights_delta

    def fit(self, input_images, goal_output, alpha=0.01):
        for series in range(len(input_images)):
            cut_image = NeuralNetwork.cut_image(
                input_images[series], len(self.hidden_layer[0])
            )

            kernel_layer = np.matmul(cut_image, self.hidden_layer.T)
            kernel_layer = ActivationFunctions.neural_network_activation(
                kernel_layer, self.hidden_layer_activation_function
            )
            flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

            output_layer = np.matmul(self.output_weights, flatten_kernel_layer)
            output_layer = ActivationFunctions.softmax(output_layer)
            output_delta = (
                2
                / len(output_layer)
                * np.subtract(output_layer, goal_output[series, :, np.newaxis])
            )

            kernel_delta = np.matmul(self.output_weights.T, output_delta)
            kernel_delta_reshaped = kernel_delta.reshape(kernel_layer.shape)
            kernel_delta_reshaped *= ActivationFunctions.derivative_function(
                kernel_layer, self.hidden_layer_activation_function
            )

            output_weights_delta = np.matmul(output_delta, flatten_kernel_layer.T)
            kernel_weights_delta = np.matmul(kernel_delta_reshaped.T, cut_image)

            self.output_weights = self.output_weights - alpha * output_weights_delta
            self.hidden_layer = self.hidden_layer - alpha * kernel_weights_delta

    def fit_conv_pool(
        self, input_images, goal_output, pooling_mask_size, pooling_step, alpha=0.01
    ):
        for series in range(len(input_images)):
            cut_image = NeuralNetwork.cut_image(
                input_images[series], len(self.hidden_layer[0])
            )

            kernel_layer = np.matmul(cut_image, self.hidden_layer.T)
            kernel_layer = ActivationFunctions.neural_network_activation(
                kernel_layer, self.hidden_layer_activation_function
            )

            binary = []
            pooled = []

            for col in zip(*kernel_layer):
                col = np.array(col).reshape(26, 26)

                pooled_image, binary_kernel_layer = NeuralNetwork.max_pooling(
                    col, pooling_mask_size, pooling_step
                )

                pooled.append(pooled_image)
                binary.append(binary_kernel_layer.flatten())

            pooled = np.vstack(pooled).T
            binary_kernel_layer = np.vstack(binary).T

            flatten_pooled_images = pooled.flatten()[:, np.newaxis]

            output_layer = np.matmul(self.output_weights, flatten_pooled_images)
            output_layer = ActivationFunctions.softmax(output_layer)
            output_delta = (
                2
                / len(output_layer)
                * np.subtract(output_layer, goal_output[series, :, np.newaxis])
            )

            kernel_delta = np.matmul(self.output_weights.T, output_delta)
            kernel_delta = np.repeat(kernel_delta, pooling_mask_size**2)

            kernel_delta_reshaped = kernel_delta.reshape(kernel_layer.shape)
            kernel_delta_reshaped *= binary_kernel_layer
            kernel_delta_reshaped *= ActivationFunctions.derivative_function(
                kernel_layer, self.hidden_layer_activation_function
            )

            output_weights_delta = np.matmul(output_delta, flatten_pooled_images.T)
            kernel_weights_delta = np.matmul(kernel_delta_reshaped.T, cut_image)

            self.output_weights = self.output_weights - alpha * output_weights_delta
            self.hidden_layer = self.hidden_layer - alpha * kernel_weights_delta

    @staticmethod
    def max_pooling(input_image, mask_size=2, step=2):
        image_height, image_width = input_image.shape
        end_x = image_width - mask_size + 1
        end_y = image_height - mask_size + 1

        cut_image = np.zeros((image_height // 2, image_width // 2))
        binary_image = np.zeros(input_image.shape)

        for row in range(0, end_y, step):
            for col in range(0, end_x, step):
                smaller_image = input_image[
                    row : row + mask_size, col : col + mask_size
                ]
                max_value = np.max(smaller_image)

                if max_value == 0:
                    continue

                cut_image[row // 2, col // 2] = max_value
                binary_image[row : row + mask_size, col : col + mask_size][
                    smaller_image == max_value
                ] = 1

        return cut_image, binary_image

    def test_model(self, input_images, goal_outputs):
        hit = 0

        for series in range(len(input_images)):
            cut_image = NeuralNetwork.cut_image(
                input_images[series], len(self.hidden_layer[0])
            )

            kernel_layer = np.matmul(cut_image, self.hidden_layer.T)
            kernel_layer = ActivationFunctions.neural_network_activation(
                kernel_layer, self.hidden_layer_activation_function
            )
            flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

            output_layer = np.matmul(self.output_weights, flatten_kernel_layer)
            output_layer = ActivationFunctions.softmax(output_layer)

            if np.argmax(output_layer) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(input_images)
        print(f"{np.round(avg * 100, 2)}%")

    def test_model_conv_pool(self, input_images, goal_outputs):
        hit = 0

        for series in range(len(input_images)):
            if series%2000 == 0:
                print(series)
            cut_image = NeuralNetwork.cut_image(
                input_images[series], len(self.hidden_layer[0])
            )

            kernel_layer = np.matmul(cut_image, self.hidden_layer.T)
            kernel_layer = ActivationFunctions.neural_network_activation(
                kernel_layer, self.hidden_layer_activation_function
            )

            pooled_images = []
            for column in zip(*kernel_layer):
                column = np.array(column).reshape(26, 26)
                pooled_image = NeuralNetwork.max_pooling(column, 2, 2)[0]
                pooled_images.append(pooled_image)

            pooled_images = np.vstack(pooled_images).T
            flatten_pooled_images = pooled_images.flatten()[:, np.newaxis]

            output_layer = np.matmul(self.output_weights, flatten_pooled_images)
            output_layer = ActivationFunctions.softmax(output_layer)

            if np.argmax(output_layer) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(input_images)
        print(f"{np.round(avg * 100, 2)}%")

    @staticmethod
    def combine_kernels(*args):
        return np.vstack(args)

    def add_layer(self, num_filters, filter_size, weights_range, activation_function):
        self.hidden_layer_activation_function = activation_function

        filter_height, filter_width = filter_size
        weights_min_value, weights_max_value = weights_range

        self.hidden_layer = np.random.uniform(
            low=weights_min_value,
            high=weights_max_value,
            size=(num_filters, filter_height * filter_width),
        )

    @staticmethod
    def prepare_numbers(data):
        return data / 255.0

    @staticmethod
    def prepare_number_labels(data):
        return np.eye(10)[data]


def zad1():
    input_image = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ]
    )
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    nn = NeuralNetwork()

    print(nn.convolution(input_image, mask))



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

    num_train_images = [1000]
    epochs = 10
    alpha = 0.01
    num_filters = 16
    mask_size = 3
    num_cut_images = (
        len(NeuralNetwork.cut_image(train_images[0], mask_size**2)) * num_filters
    )

    for num in num_train_images:
        output_weights = np.random.uniform(
            low=-0.1, high=0.1, size=(10, num_cut_images)
        )
        nn = NeuralNetwork(output_weights)
        nn.add_layer(num_filters, (mask_size, mask_size), (-0.01, 0.01), "relu")
        for epoch in range(epochs):
            start = time.time()
            nn.fit(train_images[:num], train_labels[:num], alpha)
            print(f"{epoch}: {time.time() - start}")

        print(f"{num} obrazków: ", end="")
        nn.test_model(test_images[:10000], test_labels[:10000])


def get_cols_output_weights(
    image, num_filters, conv_mask_size, pooling_mask_size, pooling_step
):
    nn = NeuralNetwork([])
    nn.add_layer(num_filters, (conv_mask_size, conv_mask_size), (-0.01, 0.01), "relu")

    cut_image = NeuralNetwork.cut_image(image, conv_mask_size**2)
    kernel_layer = np.matmul(cut_image, nn.hidden_layer.T)
    pooled_image = NeuralNetwork.max_pooling(
        kernel_layer, pooling_mask_size, pooling_step
    )[0]
    flatten_pooled_image = pooled_image.flatten()

    return len(flatten_pooled_image)


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

    num_train_images = [60000]
    num_filters = 16
    pooling_mask_size = 2
    pooling_step = 2
    conv_mask_size = 3
    epochs = 10

    output_weights_cols = get_cols_output_weights(
        train_images[0], num_filters, conv_mask_size, pooling_mask_size, pooling_step
    )

    for num in num_train_images:
        output_weights = np.random.uniform(
            low=-0.1, high=0.1, size=(10, output_weights_cols)
        )
        nn = NeuralNetwork(output_weights)
        nn.add_layer(
            num_filters, (conv_mask_size, conv_mask_size), (-0.01, 0.01), "relu"
        )
        for epoch in range(epochs):
            start = time.time()
            nn.fit_conv_pool(
                train_images[:num], train_labels[:num], pooling_mask_size, pooling_step
            )
            print(f"{epoch}: {time.time() - start}")

        print(f"{num} obrazkow: ", end="")
        nn.test_model_conv_pool(test_images[:10000], test_labels[:10000])


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

