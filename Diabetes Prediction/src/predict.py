from utils import *
from preprocess import *
from train import *


# Model Evaluatio (Accuracy Score on the training data)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ',training_data_accuracy)

# (Accuracy Score on the testing data)
X_test_prediction = classifier.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the testing data: ',training_data_accuracy)

# Making a Predictive System
input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
