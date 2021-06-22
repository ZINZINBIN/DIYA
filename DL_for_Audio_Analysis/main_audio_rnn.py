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
    

if __name__ == "__main__":
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_datasets(0.25, 0.2)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_rnn(input_shape, units = 64)
    
    # train model
    hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 32 * 4)
    plot_history(hist)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
    print("\nTest accuracy: ", test_acc)
    x_to_predict = x_test[100]
    y_to_predict = y_test[100]