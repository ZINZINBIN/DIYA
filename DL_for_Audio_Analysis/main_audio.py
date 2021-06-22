from preprocessing import *
from lib_func import *
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

DATASET_PATH = "./Data/genres_original" # 설정할 수 있다. 
JSON_PATH = "./Data/data_10.json" # 마찬가지로 수정할 수 있다. 
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def load_data(dataset_path):
    with open(JSON_PATH, "r+") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data Succesfully loaded")

    return X, y


def plot_history(hist):
    
    fig, axes = plt.subplots(2, figsize = (10,8))
    
    axes[0].plot(hist.history["accuracy"], 'r',  label = "train_accuracy")
    axes[0].plot(hist.history["val_accuracy"], "b", label = "val_accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc = "lower right")
    axes[0].set_title("accuracy eval")
    
    axes[1].plot(hist.history["loss"], 'r', label = "train_loss")
    axes[1].plot(hist.history["val_loss"], 'b', label = "val_loss")
    axes[1].set_ylabel("loss")
    axes[1].legend(loc = "lower right")
    axes[1].set_title("loss eval")
    
    plt.show()
    

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments = 10)
    

    
    x, y = load_data(DATASET_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
    
    # MLP model for audio genre classification

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (x.shape[1], x.shape[2])),
        tf.keras.layers.Dense(512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(
        optimizer = optimizer,
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.summary()
    tf.keras.utils.plot_model(model, to_file = "audio_classification_MLP.png")
    hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 32 * 4)
    plot_history(hist)
    
    '''
    model.save("audio_model_mlp.h5")
    model_tmp = tf.keras.models.load_model("audio_model_mlp.h5")
    model_tmp.summary()
    hist_tmp = model_tmp.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 1)
    '''
    
    # dropout, l2_lambda tunning
    
    dropout_list = [0.1, 0.3, 0.5]
    l2_lambda = [0.001, 0.01, 0.1]
    input_shape = (x.shape[1], x.shape[2])
    
    # dropout
    hist_list = []
    for dropout in dropout_list:
        model = build_MLP(input_shape, dropout, 0, lr = 0.01)
        hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 100)
        hist_list.append(hist)
        
        
    # accuracy
    fig, axes = plt.subplots(2, figsize = (10,8))
    
    for dropout, hist in zip(dropout_list, hist_list):
        label = "train_accuracy, dropout: " + str(dropout)
        axes[0].plot(hist.history["accuracy"], label = label)
        axes[0].set_ylabel("train Accuracy")
        axes[0].legend(loc = "lower right")
        axes[0].set_title("train accuracy eval")
        
        label = "valid_accuracy, dropout: " + str(dropout)
        axes[1].plot(hist.history["val_accuracy"], label = label)
        axes[1].set_ylabel("valid Accuracy")
        axes[1].legend(loc = "lower right")
        axes[1].set_title("valid accuracy eval")
        
    plt.show()
        
    # loss
    for dropout, hist in zip(dropout_list, hist_list):
        label = "loss, dropout: " + str(dropout)
        plt.plot(hist.history["loss"], label = label)
        plt.ylabel("loss")
        plt.legend(loc = "lower right")
        plt.ylim(0,10)
        plt.title("loss eval")
        
    plt.show()   
    
    # l2_lambda

    hist_list_l2 = []
    for l2 in l2_lambda:
        model = build_MLP(input_shape, 0, l2, lr = 0.01)
        hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 100)
        hist_list_l2.append(hist)
        
    # accuracy
    fig, axes = plt.subplots(2, figsize = (10,8))
    
    for l2, hist in zip(l2_lambda, hist_list_l2):
        label = "train_accuracy, l2_lambda: " + str(l2)
        axes[0].plot(hist.history["accuracy"], label = label)
        axes[0].set_ylabel("train Accuracy")
        axes[0].legend(loc = "lower right")
        axes[0].set_title("train accuracy eval")
        
        label = "valid_accuracy, dropout: " + str(dropout)
        axes[1].plot(hist.history["val_accuracy"], label = label)
        axes[1].set_ylabel("valid Accuracy")
        axes[1].legend(loc = "lower right")
        axes[1].set_title("valid accuracy eval")
        
    plt.show()
        
    # loss
    for l2, hist in zip(l2_lambda, hist_list_l2):
        label = "loss, l2_lambda: " + str(l2)
        plt.plot(hist.history["loss"], label = label)
        plt.ylabel("loss")
        plt.ylim(0,10)
        plt.legend(loc = "lower right")
        plt.title("loss eval")
        
    plt.show()           
    
    

    
    '''
    # CNN model for audio genre classification
    # spectogram -> (time, freq, pixel: amplitude for mfcc), 
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_datasets(0.25, 0.2)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(
        optimizer = optimizer,
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )  
    
    # train model
    hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 32 * 4)
    plot_history(hist)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
    print("\nTest accuracy: ", test_acc)
    x_to_predict = x_test[100]
    y_to_predict = y_test[100]
    
    predict(model, x_to_predict, y_to_predict)
    
    '''

'''
- Data augmentation: time stretching / pitch scaling
- dropout: increase robustness of neural network
- randomly drop neurons for same probability(0.1 - 0.5)
- regularization: add penalty to error function(constraints consideration)
- punish large weights: L1 and L2
- L1: minimize absolute value of weights(Robust to outliers), generate simple model
- L2: minimize squared value of weights(Not Robust to outlier), learn complex patterns

'''


