# train_model_tuning.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# Load CIFAR-100 dataset
def load_cifar100_data():
    with open("cifar-100/train", 'rb') as fo:
        train_data = pickle.load(fo, encoding='bytes')
    with open("cifar-100/test", 'rb') as fo:
        test_data = pickle.load(fo, encoding='bytes')

    x_train = train_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(train_data[b'fine_labels'])
    x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_data[b'fine_labels'])

    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


# Define the hyperparameter-tuned model-building function
def build_model(hp):
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

    # Define the optimizer with learning rate tuning
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Hyperparameter tuning setup
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of different hyperparameter configurations to try
    executions_per_trial=1,
    directory='tuner_results',
    project_name='cifar100_tuning'
)

# Load data
(x_train, y_train), (x_test, y_test) = load_cifar100_data()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Run the hyperparameter search
tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Retrieve the best model and save it
best_model = tuner.get_best_models(num_models=1)[0]
os.makedirs('models', exist_ok=True)
best_model.save("models/cifar100_tuned_model_v2.h5")
print("Tuned model trained and saved successfully.")
