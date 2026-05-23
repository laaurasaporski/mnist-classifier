import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_data(path_imgs, path_labels):
    X = read_idx(path_imgs)
    y = read_idx(path_labels)

    X = X.astype('float32')
    X /= 255

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123
    )

    return X_train, y_train, X_test, y_test


def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    return model


# Função para treinar o modelo
def train_model(model, train_data, test_data, batch_size=32, epochs=5):

    X_train, y_train = train_data
    X_test, y_test = test_data

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        shuffle=True
    )

    return history


# Caminhos do dataset
imgs = 'dataset/train-images.idx3-ubyte'
labels = 'dataset/train-labels.idx1-ubyte'

# Carrega os dados
X_train, y_train, X_test, y_test = load_data(imgs, labels)

# Cria o modelo
model = create_model(
    input_shape=(28, 28),
    num_classes=10
)

# Mostra arquitetura
model.summary()

# Treina o modelo
trained_model = train_model(
    model,
    (X_train, y_train),
    (X_test, y_test),
    batch_size=32,
    epochs=5
)