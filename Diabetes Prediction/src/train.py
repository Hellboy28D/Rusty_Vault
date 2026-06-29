from turtle import st
from preprocess import *

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Training then model
classifier = svm.SVC(kernel = 'linear')

# Training the Support Vector Machine Classifier
classifier.fit(X_train, Y_train)
