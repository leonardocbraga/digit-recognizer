from load_data import load_data

import model
import csv as csv
import cPickle

#retrieving the model
clf = model.get_model()

#loading the test set
X_test = load_data(test = True)[0]

#predicting the final result
output = model.predict(clf, X_test)

ids = range(1, X_test.shape[0] + 1)

#generating the final file
predictions_file = open("prediction.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId", "Label"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
