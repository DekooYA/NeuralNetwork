import numpy as np
from tensorflow.keras.datasets import mnist

def load_dataset():
    
    (x_train, y_train), _ = mnist.load_data()
    
    x_train = x_train.astype("float32") / 255
    
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    
    y_train = np.eye(10)[y_train]
    
    return x_train, y_train
