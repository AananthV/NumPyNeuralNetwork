import numpy as np
from tqdm import tqdm # For showing training progress

#########################################
## Class which represents one layer in the neural network
## Functions: init, forward, backward
#########################################
class Layer:

    #########################################
    # Function: __init__
    # Description: Function to initialise a layer, its weights and set activation and learning rate
    # Inputs:   num_inputs - Number of neurons in the previous layer
    #           num_outputs - Number of neurons in this layer
    #           activation - An object (dict) containing the activation function
    #                        and its gradient.
    #                        Format:
    #                           {
    #                               'activation': activation_function,
    #                               'gradient': gradient_of_activation_function
    #                           }
    #           learning_rate - Rate of updation of weights and biases (alpha)
    #########################################
    def __init__(self, num_inputs, num_outputs, activation, learning_rate):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.learning_rate = learning_rate

        # Initialise the wrights and biases
        self.weights = np.random.rand(num_inputs, num_outputs)
        self.biases = np.zeros(num_outputs)

    #########################################
    # Function: forward
    # Description: Performs forward propogation. Inputs the neurons of the
    #              previous layer and computes this layer.
    # Inputs: previous - numpy array (1 x n) containing the activation values of
    #                    the neurons of the previous layer
    # Outputs: activated - numpy array (1 x n) containing the activation values
    #                      of the neurons of this layer
    #########################################
    def forward(self, previous):

        # Calculate Dense values
        self.pre_activation = np.dot(previous, self.weights) + self.biases

        # Activate the neurons
        self.activated = self.activation['function'](self.pre_activation)

        return self.activated

    #########################################
    # Function: backward
    # Description: Performs backpropogation and updates its weights and biases
    # Inputs: prev_layer - numpy array (1 x n) containing the dense values of
    #                      the neurons of the previous layer
    #         prev_activations - numpy array (1 x n) containing the activation
    #                            values of the neurons of the previous layer
    #         deltas - numpy array (1 x n) containing this layer's delta values
    #         batchsize - size of chunks in which the dataset is split into
    # Outputs: n_deltas - numpy array (1 x n) containing the delta values of
    #                     the previous layer (next layer in backpropogation)
    #########################################
    def backward(self, prev_layer, prev_activations, deltas, batchsize = 32):

        # Calculate the deltas of the next layer
        n_deltas = np.dot(deltas, self.weights.T) * self.activation['gradient'](prev_layer)

        # Update weights and biases
        self.weights = self.weights + (1 / batchsize) * self.learning_rate * np.dot(prev_activations.T, deltas)
        self.biases = self.biases + (1 / batchsize) * self.learning_rate * np.sum(deltas, axis = 0)

        return n_deltas

#########################################
## Class which represents the neural network
## Functions: init, iterateBatches, train, predict, test
#########################################
class NeuralNetwork:

    #########################################
    # Function: __init__
    # Description: Function to initialise a neuralnetwork
    # Inputs: layers - A List containing layer objects (dicts).
    #                  Layer Object Format
    #                  {
    #                       'inputs': Number of neurons in the previous layer
    #                       'outputs': Number of neurons in this layer
    #                       'activation': An object (dict) containing the activation function
    #                                     and its gradient. (Refer above for format)
    #                       'learning_rate': Rate of updation of weights and biases (alpha)
    #                  }
    #########################################
    def __init__(self, layers):

        self.layers = []

        for layer in layers:

            # Create and add layers
            self.layers.append(
                Layer(layer['inputs'], layer['outputs'], layer['activation'], layer['learning_rate'])
            )

    #########################################
    # Function: iterateBatches
    # Description: Split the dataset into chunks and return an iterator
    # Inputs: X - Inputs of the dataset
    #         Y - Corresponding outputs of the dataset
    #         batchsize - Size of each chunk
    #########################################
    def iterateBatches(self, X, Y, batchsize):

        indices = np.random.permutation(len(X))

        for start_idx in tqdm(range(0, len(X) - batchsize + 1, batchsize)):

            batch = indices[start_idx:start_idx + batchsize]

            yield X[batch], Y[batch]

    #########################################
    # Function: train
    # Description: Trains the neuralnetwork on the given dataset
    # Inputs: X - Inputs of the dataset
    #         Y - Corresponding outputs of the dataset
    #         num_epochs - number of epochs of training
    #         batch_size - Size of each chunk to split the dataset into
    #########################################
    def train(self, X, Y, num_epochs = 25, batch_size = 32):

        for epoch in range(num_epochs):

            print(f"Epoch {epoch + 1}")

            for X_batch, Y_batch in self.iterateBatches(X, Y, batch_size):

                for i in range(len(X_batch)):

                    # Set intial input to the input values.
                    next_input = X_batch[i][np.newaxis]

                    # Forward propogation
                    for layer in self.layers:

                        next_input = layer.forward(next_input)

                    # Set delta of last layer to Y[i] - prediction
                    next_deltas = Y_batch[i][np.newaxis] - next_input

                    # backpropogation to train the network
                    for l in range(len(self.layers)):
                        next_deltas = self.layers[len(self.layers) - l - 1].backward(
                            self.layers[len(self.layers) - l - 2].pre_activation,
                            self.layers[len(self.layers) - l - 2].activated,
                            next_deltas,
                            batch_size
                        )

    #########################################
    # Function: predict
    # Description: Predicts the output of a given input using the NN
    # Inputs: X - Input to predict output of
    # Outputs: prediction - The Predicted output
    #########################################
    def predict(self, X):

        next_input = X[np.newaxis]

        # Forward propogate and find the output
        for layer in self.layers:

                next_input = layer.forward(next_input)

        # Required output is the last layer's activations
        return next_input

    #########################################
    # Function: test
    # Description: Tests the neural network on a test dataset
    # Inputs: X - Inputs of the dataset
    #         Y - Corresponding outputs of the dataset
    # Outputs: predictions - Predictions by the neural network
    #########################################
    def test(self, X, Y):

        squared_error = 0

        predictions = []

        for i in range(len(X)):

            predictions.append(self.predict(X[i]))

            # Using a squared error loss function
            squared_error += np.sum((Y[i] - predictions[i])**2)

        # Print Root Mean Squared Error
        RMS_Error = (squared_error / len(X))**0.5
        print("RMS Error: ", RMS_Error)

        return np.array(predictions)
