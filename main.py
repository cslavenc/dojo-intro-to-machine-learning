# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:43:36 2024

@author: slaven.cvijetic
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


if __name__ == '__main__':
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Support Vector Machine (SVM): Finds hyperplane maximizing margin between classes
    # Parameters explanation for SVM:
    # - kernel='rbf': Uses Radial Basis Function kernel
    # - C=1: Regularization parameter controlling misclassification error
    # - gamma='scale': Kernel coefficient automatically scaled based on feature range
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Accuracy:", svm_model.score(X_test, y_test))


    # Random Forest Classifier: Ensemble method combining multiple decision trees
    # Parameters explanation for Random Forests:
    # - n_estimators=100: Number of trees in the forest
    # - random_state=42: Ensures reproducibility of tree initialization

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Accuracy:", rf_model.score(X_test, y_test))
    
    
    # Extreme Gradient Boosting (XGBoost): Gradient boosting framework with internal regularization
    # Parameters explanation for XGBoost:
    # - max_depth: Maximum depth of individual trees
    # - learning_rate: Step size at each iteration during training
    # - n_estimators: Number of boosted trees
    # - subsample: Fraction of samples to be used for fitting individual trees
    # - colsample_bytree: Fraction of columns to be randomly sampled for each tree
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Accuracy:", xgb_model.score(X_test, y_test))
    

    # Gaussian Naive Bayes: Probabilistic classifier assuming multivariate Gaussian distribution
    # Parameters explanation for Naive Bayes:
    # - var_smoothing: Smoothing factor to prevent perfect fit on noise in training data
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print("Naive Bayes Accuracy:", nb_model.score(X_test, y_test))
    

    # Multi-layer Perceptron (MLP): Feedforward neural network with multiple layers of interconnected nodes
    # Parameters explanation for Neural Network:
    # - Dense(64): First hidden layer with 64 neurons
    # - activation='relu': Rectified Linear Unit activation function
    # - optimizer=Adam(learning_rate=0.001): Adam optimizer with learning rate 0.001
    # - loss='sparse_categorical_crossentropy': Loss function for multi-class classification
    # - epochs=50: Number of training iterations
    # - batch_size=32: Size of mini-batches during training
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(4,), kernel_initializer='he_normal'),
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dense(3, activation='softmax', kernel_initializer='glorot_uniform')
    ])
    nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    y_pred_nn = nn_model.predict(X_test)
    print("Neural Network Accuracy:", np.mean(np.argmax(y_pred_nn, axis=1) == y_test))
    
