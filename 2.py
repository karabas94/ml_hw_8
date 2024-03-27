import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow import keras
from keras import layers
import mlflow.keras

"""
2. побудувати модель класифікації для датасету https://keras.io/api/datasets/fashion_mnist/
модель кастомна/ленет. можна вчити на невеликій кількості епох, якщо тренування займає багато часу. 
для мене головне правильна архітектура і налаштування експерименту
"""
input_shape = (28, 28, 1)
num_classes = 10

# load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Test Data: {}'.format(x_test.shape))

# scale images to the [0,1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
        layers.Dense(num_classes, activation='softmax')
    ]
)

print(model.summary())

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# visualization with mlflow
mlflow.keras.autolog()

# train model
history = model.fit(x_train, y_train, epochs=5, batch_size=4, validation_split=0.2)

# testing model with test data set
score = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {score[0]}')
print(f'Test Accuracy: {score[1]}')
