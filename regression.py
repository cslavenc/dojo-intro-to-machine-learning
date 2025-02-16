# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:43:36 2024

@author: slaven.cvijetic
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
# classical machine learning dependencies
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# helpful functions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
# neural network dependencies
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam


### TODO : add a way to load different datasets from kaggle
if __name__ == '__main__':
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred_lm = lm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_lm)
    print("Mean Squared Error (MSE) for LM: " + str(mse))


    # Support Vector Machine (SVM): Finds hyperplane maximizing margin between classes
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # Parameters explanation for SVM:
    # - kernel='rbf': Uses Radial Basis Function kernel
    # - C=1: Regularization parameter controlling misclassification error
    # - gamma='scale': Kernel coefficient automatically scaled based on feature range
    svr_model = SVR(kernel='rbf', C=1, gamma='scale')
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    print("SVR Accuracy:", svr_model.score(X_test, y_test))


    # Random Forest Classifier: Ensemble method combining multiple decision trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # Parameters explanation for Random Forests:
    # - n_estimators=100: Number of trees in the forest
    # - random_state=42: Ensures reproducibility of tree initialization
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Accuray:", rf_model.score(X_test, y_test))
    
    
    # Extreme Gradient Boosting (XGBoost): Gradient boosting framework with internal regularization
    # https://xgboost.readthedocs.io/en/latest/
    # Parameters explanation for XGBoost:
    # - max_depth: Maximum depth of individual trees
    # - learning_rate: Step size at each iteration during training
    # - n_estimators: Number of boosted trees
    # - subsample: Fraction of samples to be used for fitting individual trees
    # - colsample_bytree: Fraction of columns to be randomly sampled for each tree
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Accuracy:", xgb_model.score(X_test, y_test))
    
    

    # Multi-layer Perceptron (MLP): Feedforward neural network with multiple layers of interconnected nodes
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    # Parameters explanation for Neural Network:
    # - Dense(64): First hidden layer with 64 neurons
    # - activation='relu': Rectified Linear Unit activation function
    # - optimizer=Adam(learning_rate=0.001): Adam optimizer with learning rate 0.001
    # - loss='sparse_categorical_crossentropy': Loss function for multi-class classification
    # - epochs=50: Number of training iterations
    # - batch_size=32: Size of mini-batches during training
    # nn_model = Sequential([
    #     Dense(64, activation='relu', input_shape=(4,), kernel_initializer='he_normal'),
    #     Dense(32, activation='relu', kernel_initializer='he_normal'),
    #     Dense(1)
    # ])
    
    # # Scale features since neural networks use data between [0,1], sometimes [-1,1] depending on the actication function
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_scaled_test = scaler.transform(X_test)
    
    # nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])
    # nn_model.fit(X_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_scaled_test, y_test))
    # y_pred_nn = nn_model.predict(X_scaled_test)
    # print("Neural Network Accuracy:", np.mean(y_pred_nn == y_test))
    
    
    # Create confusion matrices
    plt.figure(figsize=(12, 6))
    preds = [y_pred_lm, y_pred_svr, y_pred_rf, y_pred_xgb]
    models = ['Linear Model', 'SVR', 'Random Forest', 'XGBoost']
    accuracies = [mean_squared_error(y_test, pred) for pred in preds]

    # Bar plot comparing performance
    plt.figure(figsize=(12, 6))
    x = np.arange(len(preds))
    width = 0.8 / len(preds)
    
    for i, model in enumerate(models):
        y = accuracies[i]
        plt.bar(x[i], y, width, label=model, alpha=0.7)
    
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()