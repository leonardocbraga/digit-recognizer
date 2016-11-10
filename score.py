from load_data import load_data
import model as model
import numpy as np
import matplotlib.pyplot as plt

#loading training data
X, y = load_data();

data_size = X.shape[0]
limit = int(0.7*data_size)

#breaking down the training data
X_train = X[0:limit, 0::]
y_train = y[0:limit, 0::]

X_test = X[limit::, 0::]
y_test = y[limit::, 0::]

#fitting
clf = model.fit(X_train, y_train)

print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

prediction = model.predict(clf, X_test)
expected = np.argmax(y_test, axis = 1)

print "Total errors: %d" % (prediction != expected).sum()

#displaying all digits badly predicted
count = 0
shape = 28
width = 20

image = np.empty((0, shape*width))
line = np.empty((shape, 0))

for i in xrange(0, expected.shape[0]):
    if prediction[i] != expected[i]:
        count = count + 1
        line = np.hstack([line, np.reshape(X_test[i, 0::], (shape, shape))])
        if count % 20 == 0:
            image = np.vstack([image, line])
            line = np.empty((shape, 0))

if line.shape[1] != 0:
    line = np.hstack([line, np.zeros((shape, shape*width - line.shape[1]))])
    image = np.vstack([image, line])

plt.imshow(image, cmap='Greys_r')
plt.show()
