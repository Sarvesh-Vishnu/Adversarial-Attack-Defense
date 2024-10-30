# train_model.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load CIFAR-100 dataset
def load_cifar100_data():
    # Load training data
    with open("cifar-100/train", 'rb') as fo:
        train_data = pickle.load(fo, encoding='bytes')
    # Load test data
    with open("cifar-100/test", 'rb') as fo:
        test_data = pickle.load(fo, encoding='bytes')

    # Extract images and labels
    x_train = train_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(train_data[b'fine_labels'])
    x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_data[b'fine_labels'])

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


# Build CNN Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Load data, train, and save the model
(x_train, y_train), (x_test, y_test) = load_cifar100_data()
model = build_model()
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test),callbacks=[early_stopping])



# Save model to the models directory
os.makedirs('models', exist_ok=True)
model.save("models/cifar100_model.h5")
print("Model trained and saved successfully.")
