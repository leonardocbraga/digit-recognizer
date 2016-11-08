from sklearn.neural_network import MLPClassifier
from load_data import load_data

import numpy as np
import cPickle

def fit(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(450, 300, 150), max_iter=150, alpha=1e-2, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)

    print 'Training...'
    clf.fit(X, y)
	
    return clf

def predict(classifier, X):
    print 'Predicting...'
    prediction = classifier.predict(X)

    return np.argmax(prediction, axis = 1)

def generate_model():
    X, y = load_data()

    clf = fit(X, y)

    with open('model.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

def get_model():
    with open('model.pkl', 'rb') as fid:
        clf = cPickle.load(fid)
    return clf
