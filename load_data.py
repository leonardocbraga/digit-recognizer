import pandas as pd
import numpy as np
from sklearn.utils import shuffle

TRAINING_FILE_NAME = '../data/train.csv'
TEST_FILE_NAME = '../data/test.csv'

def load_data(test = False):
    data_frame = pd.read_csv(TRAINING_FILE_NAME if not test else TEST_FILE_NAME, header=0).values
    first_index = 1 if not test else 0
    X = data_frame[0::, first_index::] / 255.
    y = np.empty(0)

    if not test:
        data_size = data_frame[0::, 0].size
        y = np.zeros((data_size, 10))
        for index in xrange(data_size):
            label = data_frame[index, 0]
            y[index, label] = 1

        X, y = shuffle(X, y, random_state = 0)
    return X, y