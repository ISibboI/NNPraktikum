# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import logging
import random

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

    def train(self, verbose=True, debug=False):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        bestClassificationResult = len(self.validationSet.input)
        stagnation = 0

        for i in range(0, self.epochs):
            # Minibatch update
            # Doesn't help in this case, since the classifier is linear, but I implemented it anyways
            shuffledIndexes = range(0, len(self.trainingSet.input))
            random.shuffle(shuffledIndexes)
            batchSize = 64
            
            for offset in range(0, len(self.trainingSet.input), batchSize):
                batchData = []
                batchLabels = []
            
                for index in range(offset, min(offset + batchSize, len(shuffledIndexes))):
                    index = shuffledIndexes[index]
                    batchData.append(self.trainingSet.input[index])
                    batchLabels.append(self.trainingSet.label[index])
                
                self.batchUpdate(batchData, batchLabels, float(batchSize) / len(self.trainingSet.input))

            # Validation
            validationClasses = self.evaluate(self.validationSet.input)
            error = 0

            for j in range(0, len(validationClasses)):
                if validationClasses[j] != (self.validationSet.label[j] != 0):
                    error += 1

            if verbose:
                print("Validation accuracy: {0}%".format(100 - error / float(len(validationClasses)) * 100))

            # Abort early if classification results stop getting better to prevent overfitting
            if error >= bestClassificationResult:
                stagnation += 1

                if stagnation >= 5:
                    break
            else:
                bestClassificationResult = error
                stagnation = 0
                
    def batchUpdate(self, trainingData, trainingLabels, learningRateModifier=1.0, debug=False):
        classifications = self.evaluate(trainingData)
        errors = []
        
        # Find wrong classifications
        for j in range(0, len(classifications)):
            if classifications[j] != (trainingLabels[j] != 0):
                errors.append(j)

        if debug:
            print("Found {0}/{1} errorneous classifications".format(len(errors), len(trainingData)))

        # Update weights
        for j in errors:
            for k in range(0, len(self.weight)):
                self.weight[k] -= float(trainingData[j][k]) * self.learningRate * learningRateModifier / len(trainingData)

        if debug:
            print("Sum of abs weights: {0}".format(sum(map(abs, self.weight))))

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) > 0

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
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight))
