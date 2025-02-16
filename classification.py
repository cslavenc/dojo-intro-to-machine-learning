# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:43:36 2024

@author: slaven.cvijetic
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# classical machine learning dependencies
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
# helpful functions
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
# neural network dependencies
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam


def preparePasswordDataset():
    # This is a good candidate for further feature engineering
    train = pd.read_csv(os.getcwd()+"\\data\\classification\\passwords.csv")
    test = pd.read_csv(os.getcwd()+"\\data\\classification\\passwords_test.csv")
    # Add a length column to calculate the length of the text
    train["length"] = train["password"].apply(lambda x : len(x))
    test["length"] = test["password"].apply(lambda x : len(x))
    X_train = train["length"].to_numpy().reshape(-1, 1)
    y_train = train["strength"].to_numpy()
    X_test = test["length"].to_numpy().reshape(-1, 1)
    y_test = test["strength"].to_numpy()
    return X_train, X_test, y_train, y_test


def prepareMaintenanceDataset():
    df = pd.read_csv(os.getcwd()+"\\data\\classification\\maintenance.csv")
    labelEncoder = LabelEncoder()
    X = pd.get_dummies(df.iloc[:,:-1]).to_numpy()
    y = labelEncoder.fit_transform(df.iloc[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    # load dataset
    X_train, X_test, y_train, y_test = prepareMaintenanceDataset()
    
    # Support Vector Machine (SVM): Finds hyperplane maximizing margin between classes
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # Parameters explanation for SVM:
    # - kernel='rbf': Uses Radial Basis Function kernel
    # - C=1: Regularization parameter controlling misclassification error
    # - gamma='scale': Kernel coefficient automatically scaled based on feature range
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Accuracy:", svm_model.score(X_test, y_test))


    # Random Forest Classifier: Ensemble method combining multiple decision trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # Parameters explanation for Random Forests:
    # - n_estimators=100: Number of trees in the forest
    # - random_state=42: Ensures reproducibility of tree initialization
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Accuracy:", rf_model.score(X_test, y_test))
    
    
    # Extreme Gradient Boosting (XGBoost): Gradient boosting framework with internal regularization
    # https://xgboost.readthedocs.io/en/latest/
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
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    # Parameters explanation for Naive Bayes:
    # - var_smoothing: Smoothing factor to prevent perfect fit on noise in training data
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print("Naive Bayes Accuracy:", nb_model.score(X_test, y_test))
    

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
    #     Dense(3, activation='softmax')
    # ])
    
    # # Scale features since neural networks use data between [0,1], sometimes [-1,1] depending on the actication function
    # scaler = StandardScaler()
    # scaler = scaler.fit(np.concatenate((X_train, X_test)))
    # X_scaled = scaler.transform(X_train)
    # X_scaled_test = scaler.transform(X_test)
    
    # nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # nn_model.fit(X_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_scaled_test, y_test))
    # y_pred_nn = np.argmax(nn_model.predict(X_scaled_test), axis=1)
    # print("Neural Network Accuracy:", np.mean(y_pred_nn == y_test))
    
    
    # Create confusion matrices
    plt.figure(figsize=(12, 6))
    preds = [y_pred_svm, y_pred_rf, y_pred_xgb, y_pred_nb]
    models = ['SVM', 'Random Forest', 'XGBoost', 'Naive Bayes']
    accuracies = [accuracy_score(y_test, pred) for pred in preds]
    
    for i, pred in enumerate(preds):
        cm = confusion_matrix(y_test, pred)
        plt.subplot(1, len(models), i+1)
        plt.imshow(cm, interpolation='nearest')
        plt.title(f'Confusion Matrix - {i}')
        plt.colorbar()
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

    plt.tight_layout()
    plt.show()

    # Bar plot comparing accuracy
    plt.figure(figsize=(12, 6))
    x = np.arange(len(preds))
    width = 0.8 / len(preds)
    
    for i, model in enumerate(models):
        y = accuracies[i]
        plt.bar(x[i], y, width, label=model, alpha=0.7)
    
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()