import sys
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# Load data
df = pd.read_csv('D:\\Project\\fer2013.csv')

X_train, train_y, X_test, test_y = [], [], [], []

for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occurred at index: {index} and row: {row}")

# Convert to numpy arrays
X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

# Reshape and normalize data
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) / 255.0
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1) / 255.0

# Convert labels to one-hot encoding
num_labels = 7
train_y = to_categorical(train_y, num_classes=num_labels)
test_y = to_categorical(test_y, num_classes=num_labels)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True)

# Define the model with increased complexity and regularization
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_labels, activation='softmax')
])

# Implement Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

# Compile the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Adjust Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(datagen.flow(X_train, train_y, batch_size=64),
                    epochs=100,  # Increased epochs
                    steps_per_epoch=len(X_train) / 64,
                    validation_data=(X_test, test_y),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Saving the model
model.save("fer.h5")
