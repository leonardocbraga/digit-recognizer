from load_data import load_data
import model as model
import numpy as np
import matplotlib.pyplot as plt

X, y = load_data();

data_size = X.shape[0]
limit = int(0.7*data_size)

X_train = X[0:limit, 0::]
y_train = y[0:limit, 0::]

X_test = X[limit::, 0::]
y_test = y[limit::, 0::]

clf = model.fit(X_train, y_train)

print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

prediction = model.predict(clf, X_test)
expected = np.argmax(y_test, axis = 1)

print "Total errors: %d" % (prediction != expected).sum()

for i in xrange(100, expected.shape[0]):
    if prediction[i] != expected[i] and i < 200:
        plt.figure()
        plt.imshow(np.reshape(X_test[i, 0::], (28, 28)), cmap='Greys_r')
        plt.title('Predicted: %d; Original: %d' % (prediction[i], expected[i]))

plt.show()
