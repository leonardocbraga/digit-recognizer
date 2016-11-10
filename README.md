# Project

This task refers to the [Kaggle competition](https://www.kaggle.com/c/digit-recognizer/) to classify handwritten digits using the famous MNIST data. It consists of files to build the model responsible for fitting and predicting, load data from training and test set and generate the submission file. Also, there is a file to test how well the classifier chosen is doing on 30% of the training set after fitting the other 70%. The digits that are badly predicted are displayed altogether.

# Platform

The source code is written in [Python] (https://www.python.org/) with [scikit-learn] (http://scikit-learn.org/) whereas they can afford Machine Learning Algorithms.

# Data
The training and test sets can be obtained at [the challenge data page](https://www.kaggle.com/c/digit-recognizer/data/). The training and the test set are lists of 42000 and 28000 gray-scale images of hand-drawn digits respectively.

# Solution

The classifier used to fit the data was [Multi-layer Perceptron Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) from scikit-learn package. This is a Neural Network with five hidden layers to train examples with 784 features, each of them representing a pixel in gray-scale of an image which is 28 pixels in height and 28 pixels in width.

# Score

Getting it done gives a score of 0.97829 on the [Public Leaderboard](https://www.kaggle.com/c/digit-recognizer/leaderboard) indicating that the model correctly classified all but 3% of the images.
