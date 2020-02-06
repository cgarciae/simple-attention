from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from plotly import express as px
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler

from .tf_keras_transformer import MultiHeadSelfAttention


def main(
    data_path: Path = typer.Option(...),
    n_layers: int = 1,
    n_neurons: int = 16,
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 0.001,
    viz: bool = False,
    dp_percentage: float = 0.4,
) -> None:

    train_path = data_path / "training-set.csv"
    test_path = data_path / "test-set.csv"

    df_train = pd.read_csv(train_path, names=["x1", "x2", "label"])
    df_test = pd.read_csv(test_path, names=["x1", "x2", "label"])

    X_train = df_train[["x1", "x2"]].to_numpy().astype(np.float32)
    y_train = df_train["label"].to_numpy().astype(int)

    X_test = df_test[["x1", "x2"]].to_numpy().astype(np.float32)
    y_test = df_test["label"].to_numpy().astype(int)

    transform = StandardScaler()

    X_train = transform.fit_transform(X_train)
    X_test = transform.transform(X_test)

    N = len(X_train)
    idx = np.random.choice(N, int(N * dp_percentage))

    model = SimpleMemory(
        x=X_train[idx], y=y_train[idx], n_layers=n_layers, n_neurons=n_neurons,
    )
    model(X_train[:2])
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(lr=lr),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.BinaryAccuracy()],
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )
    model.summary()

    if viz:

        # plain scatter
        sizes = np.ones((N,), dtype=np.float32) * 9
        sizes[idx] = 12
        opacity = np.ones((N,), dtype=np.float32)
        fig = go.Figure(
            [
                go.Scatter(
                    x=X_train[:, 0],
                    y=X_train[:, 1],
                    marker=go.scatter.Marker(
                        color=y_train,
                        size=sizes,
                        line_width=2,
                        line_color="DarkSlateGrey",
                        opacity=opacity,
                    ),
                    mode="markers",
                )
            ]
        )
        fig.update_layout(template="simple_white")
        fig.show()

        # only db

        sizes = np.ones((N,), dtype=np.float32) * 9
        sizes[idx] = 12
        opacity = np.ones((N,), dtype=np.float32) * 0.01
        opacity[idx] = 1
        fig = go.Figure(
            [
                go.Scatter(
                    x=X_train[:, 0],
                    y=X_train[:, 1],
                    marker=go.scatter.Marker(
                        color=y_train,
                        size=sizes,
                        line_width=2,
                        line_color="DarkSlateGrey",
                        opacity=opacity,
                    ),
                    mode="markers",
                )
            ]
        )
        fig.update_layout(template="simple_white")
        fig.show()

        # decision boundary
        sizes = np.ones((N,), dtype=np.float32) * 9
        sizes[idx] = 12
        opacity = np.ones((N,), dtype=np.float32)
        fig = go.Figure(
            [
                go.Scatter(
                    x=X_train[:, 0],
                    y=X_train[:, 1],
                    marker=go.scatter.Marker(
                        color=y_train,
                        size=sizes,
                        line_width=2,
                        line_color="DarkSlateGrey",
                        opacity=opacity,
                    ),
                    mode="markers",
                )
            ]
        )
        fig.update_layout(template="simple_white")
        xx, yy, zz = decision_boundaries(X_train, model)
        xx = xx[0]
        yy = yy[:, 0]
        fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))
        fig.show()

        # invert points
        X_train[:, 1] *= -1

        # get original decision boundary with new points

        sizes = np.ones((N,), dtype=np.float32) * 9
        sizes[idx] = 12
        opacity = np.ones((N,), dtype=np.float32)
        fig = go.Figure(
            [
                go.Scatter(
                    x=X_train[:, 0],
                    y=X_train[:, 1],
                    marker=go.scatter.Marker(
                        color=y_train,
                        size=sizes,
                        line_width=2,
                        line_color="DarkSlateGrey",
                        opacity=opacity,
                    ),
                    mode="markers",
                )
            ]
        )
        fig.update_layout(template="simple_white")
        xx, yy, zz = decision_boundaries(X_train, model)
        xx = xx[0]
        yy = yy[:, 0]
        fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))
        fig.show()

        # with updated database
        model.x = X_train[idx]

        sizes = np.ones((N,), dtype=np.float32) * 9
        sizes[idx] = 12
        opacity = np.ones((N,), dtype=np.float32)
        fig = go.Figure(
            [
                go.Scatter(
                    x=X_train[:, 0],
                    y=X_train[:, 1],
                    marker=go.scatter.Marker(
                        color=y_train,
                        size=sizes,
                        line_width=2,
                        line_color="DarkSlateGrey",
                        opacity=opacity,
                    ),
                    mode="markers",
                )
            ]
        )
        fig.update_layout(template="simple_white")
        xx, yy, zz = decision_boundaries(X_train, model)
        xx = xx[0]
        yy = yy[:, 0]
        fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))
        fig.show()


class SimpleMemory(tf.keras.Model):
    def __init__(
        self, x, y, n_layers, n_neurons,
    ):
        super().__init__()

        self.x = x
        self.y = y

        self.f = tf.keras.Sequential()

        for _ in range(n_layers):
            self.f.add(tf.keras.layers.Dense(n_neurons, activation="elu"))
            self.f.add(tf.keras.layers.LayerNormalization())

        self.attention = tf.keras.layers.Attention()

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        xq = inputs[:, None, :]

        xdb = tf.constant(self.x)[None]
        xdb = tf.tile(xdb, [batch_size, 1, 1])

        ydb = tf.constant(self.y, dtype=tf.float32)[None, :, None]
        ydb = tf.tile(ydb, [batch_size, 1, 1])

        q = self.f(xq)
        k = self.f(xdb)
        v = ydb

        net = self.attention([q, v, k])[:, 0]

        return net


def decision_boundaries(X, model, n=20):

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    print(x_min, x_max)
    print(y_min, y_max)

    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max + hx, hx), np.arange(y_min, y_max + hy, hy)
    )

    # Obtain labels for each point in mesh using the model.
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    Z = model(points).numpy()
    Z = (Z > 0.5).astype(np.int32)

    zz = Z.reshape(xx.shape)

    return xx, yy, zz


def flatten(x):
    return sum(x, [])


if __name__ == "__main__":
    typer.run(main)
