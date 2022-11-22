import numpy as np
import tensorflow as tf
import gzip
from PIL.ImageChops import offset
from tensorflow.python.keras.models import load_model


def load_MNIST_data():
# Trainingsdaten
    with gzip.open("./train-labels-idx1-ubyte.gz") as file:
        train_labels = np.frombuffer(file.read(), np.uint8(), offset=8)
    with gzip.open("./train-images-idx3-ubyte.gz") as file:
        train_images = np.frombuffer(file.read(), np.uint8(), offset=16).reshape(len(train_labels), 28, 28) / 255.0
# Testdaten
    with gzip.open("./t10k-labels-idx1-ubyte.gz") as file:
        test_labels = np.frombuffer(file.read(), np.uint8, offset=8)
    with gzip.open("./t10k-images-idx3-ubyte.gz") as file:
        test_images = np.frombuffer(file.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28) / 255.0




    test_images = np.reshape(test_images, (-1, 28, 28, 1))
    train_images = np.reshape(train_images, (-1, 28, 28, 1))

    return train_images, train_labels, test_images, test_labels


train_images, train_labels, test_images, test_labels = load_MNIST_data()


def MNIST_model():
    model = tf.keras.models.Sequential()
    # Layer1
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    # Layer2
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model
model = MNIST_model()

model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.1)


evaluation_results = model.evaluate(test_images, test_labels)

print("Loss: {}". format(evaluation_results[0]))
#Ausgabe Loss
print("Accuracy: {}".format(evaluation_results[1]))
#Ausgabe Accuracy

model.save("MNIST.h5")
model = load_model("MNIST.h5")

#image_index = 4444
#predictions = model.predict(test_images[image_index].reshape(1, 28, 28, 1))

predictions = model.predict(test_images[50].reshape(-1, 28, 28, 1))
#print(predictions.argmax())

#print(predictions)
print(predictions.argmax())
