
import numpy as np
from numpy import exp
from sklearn.metrics import accuracy_score

# from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights
        input_size = self.training_set.input.shape[1]

        # add bias values ("1"s) at the beginning of all data sets
        # self.training_set.input = np.insert(self.training_set.input, 0, 1,
        #                                     axis=1)
        # self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
        #                                       axis=1)
        # self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        self.layers = []
        output_activation = "sigmoid"
        self.layers.append(LogisticLayer(input_size, input_size, None, output_activation, False))
        self.layers.append(LogisticLayer(input_size, input_size, None, output_activation, False))
        self.layers.append(LogisticLayer(input_size, input_size, None, output_activation, False))
        self.layers.append(LogisticLayer(input_size, 1, None, output_activation, True))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        self.layers[0].forward(inp)

        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].outp)

        # if self.layers[-1].outp[0] != 1.0:
        #     print(self.layers[-1].outp[0])

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        self.layers[-1].computeDerivative(target, None)

        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].computeDerivative(self.layers[i + 1].deltas, self.layers[i + 1].weights)

    def _update_weights(self, round_fraction):
        """
        Update the weights of the layers by propagating back the error
        """

        # Learning rate modification function
        modifier = (exp(1 - round_fraction) - 1) * 40 + 1

        for i in range(len(self.layers)):
            self.layers[i].updateWeights(self.learning_rate * modifier)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch(epoch / (self.epochs - 1))

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self, round_fraction):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self._feed_forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            self._compute_error(np.array(label))

            # Update weights in the online learning fashion
            self._update_weights(round_fraction)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        self._feed_forward(test_instance)
        return self.layers[-1].outp[0] > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """

        if test is None:
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
