import numpy as np
import scipy as sp
import os
import librosa, librosa.display
import math
import json
import tensorflow as tf
import matplotlib.pyplot as plt

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

def build_rnn(input_shape, units = 64):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, input_shape = input_shape, return_sequences = True))
    model.add(tf.keras.layers.LSTM(units))
    model.add(tf.keras.layers.Dense(units, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation = "softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(
        optimizer = optimizer,
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    model.summary()
    return model

def build_MLP(input_shape, dropout, l2_lambda, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = input_shape),
        tf.keras.layers.Dense(512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(64, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(
        optimizer = optimizer,
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.summary()
    
    return model


def load_data(dataset_path):
    with open(JSON_PATH, "r+") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data Succesfully loaded")

    return X, y

def plot_history(hist):
    
    fig, axes = plt.subplots(2)
    
    axes[0].plot(hist.history["acc"], 'r',  label = "train_accuracy")
    axes[0].plot(hist.history["val_acc"], "b", label = "val_accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc = "lower right")
    axes[0].set_title("accuracy eval")
    
    axes[1].plot(hist.history["loss"], 'r', label = "train_loss")
    axes[1].plot(hist.history["val_loss"], 'b', label = "val_loss")
    axes[1].set_ylabel("loss")
    axes[1].legend(loc = "lower right")
    axes[1].set_title("loss eval")
    
    plt.show()
    
def prepare_datasets(test_size, valid_size):
    x, y = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_size, random_state = 42)
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = valid_size, random_state = 42)
    
    # add an axis to input set (2d -> 3d)
    x_train = x_train[..., np.newaxis] 
    x_valid = x_valid[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def build_model(input_shape):
    
    model = tf.keras.models.Sequential()
    
    # 1st cnn layer
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides = (2,2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    
    # 2nd cnn layer
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides = (2,2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())  
    
    # 3rd cnn layer
    model.add(tf.keras.layers.Conv2D(32, (2,2), activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())     
    
    # flatten output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation = "softmax"))
    
    return model

def predict(model, x, y):
    x = x[np.newaxis, ...] # 4d array (1,h,w,c = 1)
    prediction = model.predict(x)
    predicted_index = np.argmax(prediction, axis = 1)
    print("Target:{}, Predict label:{}".format(y, predicted_index))