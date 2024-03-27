import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from tensorflow import keras
from keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score

"""
1. побудувати модель класифікації для датасету mnist на 2 класи: <5 і >=5. 
(тобто якщо вхідне зображення - це цифра 8, то клас=1, якщо цифра 4 - то клас=0). 
архітектуру можна побудувати з нуля, а можна взяти ле-нет (більшу не треба - буде довге навчання). 
обчислити precision, recall і f1score на тестовому датасеті
"""

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale images to the [0,1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# creating binary labels for class y>=5(1) and y<5(0)
y_train_b = (y_train >= 5).astype(int)
y_test_b = (y_test >= 5).astype(int)

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', padding="valid"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)
print(model.summary())

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train model
history = model.fit(x_train, y_train_b, epochs=5, batch_size=4)

# testing model with test data set
y_pred = (model.predict(x_test) >= 0.5).astype(int)

# metrics
precision = precision_score(y_test_b, y_pred)
recall = recall_score(y_test_b, y_pred)
f1score = f1_score(y_test_b, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1score}')
