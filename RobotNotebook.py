# %%
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging, sys
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from pylab import *

logging.disable(sys.maxsize)

TRAIN_PATH = "C:/Users/Simon B/PycharmProjects/RobotJupyter/data/X_train.csv"
TRAIN_LABEL_PATH = "C:/Users/Simon B/PycharmProjects/RobotJupyter/data/y_train.csv"
TEST_PATH = "C:/Users/Simon B/PycharmProjects/RobotJupyter/data/X_test.csv"
RESULT_PATH = "C:/Users/Simon B/PycharmProjects/RobotJupyter/data/submission.csv"

train_data = pd.read_csv(TRAIN_PATH, sep=",")
train_label = pd.read_csv(TRAIN_LABEL_PATH, sep=",")

# %%
print(train_data.shape)

# %%
print(train_data.head())

# %%
print(train_data["series_id"].nunique())

# %%
print(train_label.shape)

# %%
print(train_label.head())

# %%
print(train_label["surface"].unique())

# %%
print(train_label["group_id"].nunique())


def transform_data(dataframe):
    transformed_dataframe = []
    temp_array = []
    index = 0
    dataframe_size = dataframe.shape[0]
    for i in range(dataframe_size):
        if dataframe["series_id"][i] != index:
            index += 1
            transformed_dataframe.append(np.asarray(temp_array))
            temp_array = []

        temp_array.append([
            dataframe["orientation_X"][i],
            dataframe["orientation_Y"][i],
            dataframe["orientation_Z"][i],
            dataframe["orientation_W"][i],
            dataframe["angular_velocity_X"][i],
            dataframe["angular_velocity_Y"][i],
            dataframe["angular_velocity_Z"][i],
            dataframe["linear_acceleration_X"][i],
            dataframe["linear_acceleration_Y"][i],
            dataframe["linear_acceleration_Z"][i]
        ])

        if i == dataframe_size - 1:
            transformed_dataframe.append(np.asarray(temp_array))

    return np.asarray(transformed_dataframe)


# %%
transformed_data = transform_data(train_data)
print(transformed_data)
print(transformed_data.shape)

# %%
plt.figure(figsize=(26, 16))
for i, col in enumerate(train_data.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(train_data.loc[train_data['series_id'] == 21, col])
    plt.title(col)

# %%
encoder = LabelEncoder()

# We separate the dataset in a training and validation set to verify if our model generalizes well (no overfiting)
train_examples = transformed_data[0:3400]
train_targets = encoder.fit_transform(train_label["surface"].head(3400))

validation_examples = transformed_data[3400:3810]
validation_targets = encoder.fit_transform(train_label["surface"].tail(410))

encoder.fit_transform(train_label["surface"])

# %%
def train_model(
        training_examples,
        training_targets,
        epoch
):

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 10)),
        keras.layers.Dense(156, activation=tf.nn.relu),
        keras.layers.Dense(18, activation=tf.nn.relu),
        keras.layers.Dense(9, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(training_examples, training_targets, epochs=epoch)

    return model


def evaluate_model(model):
    test_loss, test_acc = model.evaluate(validation_examples, validation_targets)

    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)

# %%
# Play with these (you can also try to change the architecture of the DNN) and try to consistently get a good accuracy,
# auc, precision and recall score (mostly accuracy in this case)
deep_classifier = train_model(
    training_examples=train_examples,
    training_targets=train_targets,
    epoch=100
)

# %%
evaluate_model(
    model=deep_classifier
)

# %%
test_data = pd.read_csv(TEST_PATH, sep=",")
transformed_test_data = transform_data(test_data)

# %%
print(test_data["series_id"].unique())

# %%
results = deep_classifier.predict(transformed_test_data)
results = [np.argmax(result) for result in results]

# %%
submission = pd.DataFrame({
    "series_id": test_data["series_id"].unique(),
    "surface": encoder.inverse_transform(results)
})

submission.to_csv(RESULT_PATH, index=False)
