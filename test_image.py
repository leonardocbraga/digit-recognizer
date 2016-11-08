import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
import model
import cPickle

with open('model.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

X_test = load_data(test = True)[0]

output = model.predict(X_test, clf)

for i in xrange(0, 80):
    if output[i] != 9:
        continue
    plt.figure()
    plt.imshow(np.reshape(X_test[i, 0::], (28, 28)), cmap='Greys_r')
    plt.title('Digit predicted: %d' % output[i])

plt.show()
