import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def lennard_jones_data():
    lj = lambda r: 4 * ((1.0 / r) ** (12) - (1.0 / r) ** 6)

    x = np.linspace(0.90, 3.0, 10000).reshape(-1, 1)
    y = lj(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = lennard_jones_data()

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(50, activation="tanh")
        self.d2 = Dense(10, activation="tanh")
        self.d3 = Dense(1, activation="linear")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return x


model = MyModel()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
test_loss = tf.keras.metrics.Mean(name="test_loss")


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)


epochs = 50

for epoch in range(epochs):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = "Epoch {}, Loss: {}, Test Loss: {}"
    print(template.format(epoch + 1, train_loss.result(), test_loss.result()))

x_plot = np.linspace(0.90, 3.0, 1000)
y_plot = np.array(model(x_plot.reshape(-1, 1)))

x_test = x_test.reshape(-1)
y_test = y_test.reshape(-1)
ind = np.argsort(x_test)
x_test = x_test[ind]
y_test = y_test[ind]

plt.plot(x_plot, y_plot)
plt.plot(x_test, y_test)
plt.show()
