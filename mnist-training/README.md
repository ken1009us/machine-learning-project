# MNIST Handwritten Digits Classification

This code demonstrates the implementation of a neural network model using Keras to classify handwritten digits from the MNIST dataset.

## Prerequisites

Make sure you have the following libraries installed:

```bash
numpy
pandas
matplotlib
keras
```

You can install these libraries using pip.

## Usage

1. Import the required libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
```

2. Load the MNIST dataset and preprocess the data:

```python
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# Display information about the dataset
print('Train data:', len(x_train_image))
print('Test data:', len(x_test_image))
print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)
print('x_test_image:', x_test_image.shape)
print('y_test_label:', y_test_label.shape)

# Plot an example image
plt.imshow(x_train_image[0], cmap='binary')
plt.show()

# Preprocess the data
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
y_train_label = to_categorical(y_train_label, 10)
y_test_label = to_categorical(y_test_label, 10)

```

3. Define and compile the model:

```python
model = Sequential()
model.add(Flatten(input_shape=x_Train.shape[1:]))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```


4. Train the model:

```python
train_history = model.fit(x=x_Train_normalize, y=y_train_label, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)
```


5. Evaluate the model:

```python
score = model.evaluate(x_Test_normalize, y_test_label, verbose=0)
accuracy = 100 * score[1]
print('The final accuracy of the model is %.4f%%' % accuracy)
```

6. Save the model:

```python
model.save('firstmodel.h5')
```

7. Visualize prediction results:

```python
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    # Plot images with labels and predictions
    # ...
    # Code for the function provided in the question

# Generate predictions for test data
prediction = model.predict(x_Test_normalize)

# Plot example images with labels and predictions
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=280)
```

## Acknowledgments

The MNIST dataset was used in this project, which is a widely used benchmark dataset for handwritten digit recognition.