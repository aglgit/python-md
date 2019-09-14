import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


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

    def derivative(self, x):
        with tf.GradientTape() as t:
            t.watch(x)
            y = self.call(x)

        return t.gradient(y, x)


def lennard_jones_data():
    lj = lambda r: 4 * ((1.0 / r) ** (12) - (1.0 / r) ** 6)

    x = np.linspace(0.90, 3.0, 10000).reshape(-1, 1)
    y = lj(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return (x_train, y_train), (x_test, y_test)


def plot_network_and_derivative(x_test, y_test, model):
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    lj_derivative = lambda r: -24 * (2 * r ** (-13) - r ** (-7))

    x_plot = x_test.copy()
    y_plot = y_test.copy()
    y_model = model(x_plot)
    dy_model = -model.derivative(tf.convert_to_tensor(x_plot))

    x_plot = np.array(x_plot).reshape(-1)
    y_plot = np.array(y_plot).reshape(-1)
    y_model = np.array(y_model).reshape(-1)
    dy_model = np.array(dy_model).reshape(-1)

    ind = np.argsort(x_plot)
    x_plot = x_plot[ind]
    y_plot = y_plot[ind]
    y_model = y_model[ind]
    dy_model = dy_model[ind]

    dy_plot = -lj_derivative(x_plot)

    plt.plot(x_plot, y_plot)
    plt.plot(x_plot, y_model)
    plt.title("Neural network potential comparison")
    plt.xlabel("Radial distance r")
    plt.ylabel("Potential energy V(r)")
    plt.legend(["Lennard-Jones", "Neural Network"])
    plt.savefig(os.path.join(plot_dir, "potential_comparison.png"))
    plt.clf()

    plt.plot(x_plot, abs(y_model - y_plot))
    plt.title("Absolute error of neural network")
    plt.xlabel("Radial distance r")
    plt.ylabel("Absolute error")
    plt.savefig(os.path.join(plot_dir, "potential_absolute_error.png"))
    plt.clf()

    plt.plot(x_plot, dy_plot)
    plt.plot(x_plot, dy_model)
    plt.title("Derivative of neural network potential")
    plt.xlabel("Radial distance r")
    plt.ylabel("Force dV(r)")
    plt.legend(["Lennard-Jones", "Neural Network"])
    plt.savefig(os.path.join(plot_dir, "force_comparison.png"))
    plt.clf()

    plt.plot(x_plot, abs(dy_model - dy_plot))
    plt.title("Absolute error of neural network force")
    plt.xlabel("Radial distance r")
    plt.ylabel("Absolute error")
    plt.savefig(os.path.join(plot_dir, "force_absolute_error.png"))
    plt.clf()

    energy_rmse = np.sqrt(np.sum((y_model - y_plot) ** 2))
    force_rmse = np.sqrt(np.sum((dy_model - dy_plot) ** 2))

    print("Energy RMSE: {}, Force RMSE: {}".format(energy_rmse, force_rmse))


(x_train, y_train), (x_test, y_test) = lennard_jones_data()

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

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


epochs = 100

for epoch in range(epochs):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = "Epoch {}, Loss: {}, Test Loss: {}"
    print(template.format(epoch + 1, train_loss.result(), test_loss.result()))

sns.set()
plot_network_and_derivative(x_test, y_test, model)
