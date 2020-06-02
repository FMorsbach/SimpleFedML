import sys

import tensorflow as tf
import json
import numpy as np
from tensorflow import keras

# TODO Implement modelling through subclassing, make KerasModel abstract


class KerasModel:
    def __init__(self):

        config, weights = loadModelFromDisk("config.txt", "weights.txt")

        self.model = keras.models.model_from_json(json.dumps(config))
        self.model.set_weights(weights)
        self.model.compile(optimizer='sgd',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']
        )
        (_, _), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

    def serialize(self):
        data = {}
        config = self.model.to_json()
        weights = [w.tolist() for w in self.model.get_weights()]
        data["config"] = json.loads(config)
        data["weights"] = weights
        return json.dumps(data)

    def evaluate(self):
        evaluation = self.model.evaluate(self.x_test,  self.y_test, verbose=2)
        return evaluation

    def aggregateUpdates(self, updates):
        weights = [0*w for w in self.model.get_weights()]

        for update in updates:
            weights = [w1 + w2 for w1, w2 in zip(weights, update[1])]

        weights = [w/len(updates) for w in weights]
        weights = [np.array(w) for w in weights]

        self.model.set_weights(weights)

    def writeToDisk(self):
        self.model.save("trained_model")


def loadModelFromDisk(c, w):
    config = loadConfigurationFromFile(c)
    weights = loadWeightsFromFile(w)

    return config, weights


def loadWeightsFromFile(file):
    try:
        with open(file, "r") as weights_file:
            weights = weights_file.read()
            weights = json.loads(weights)
            weights = [np.array(w) for w in weights]
    except IOError as e:
        print("Cant open", file, "Error:", e, file=sys.stderr)
        sys.exit(1)
    except json.decoder.JSONDecodeError as e:
        print("Cant parse", file, "to json", "Error:", e, file=sys.stderr)
        sys.exit(1)

    if weights is None:
        print("Loaded weights from", file, "are empty", file=sys.stderr)
        sys.exit(1)
    elif not checkWeightsFormat(weights):
        print("Weights do not match expected format", file=sys.stderr)
        sys.exit(1)
    else:
        print("Weights loaded successfully from", file)

    return weights


def checkWeightsFormat(weights):
    if not isinstance(weights, list):
        return False
    if not len(weights) > 0:
        return False
    if not isinstance(weights[0], np.ndarray):
        return False
    if not len(weights[0]) > 0:
        return False
    return True


def loadConfigurationFromFile(c):
    try:
        with open(c, "r") as config_file:
            config = config_file.read()
            config = json.loads(config)
    except IOError as e:
        print("Cant open", c, "Error:", e, file=sys.stderr)
        sys.exit(1)
    except json.decoder.JSONDecodeError as e:
        print("Cant parse", c, "to json", "Error:", e, file=sys.stderr)
        sys.exit(1)

    if config is None:
        print("Loaded config from", c, ", is empty", file=sys.stderr)
        sys.exit(1)
    elif not checkConfigurationFormat(config):
        print("Configuration does not match expected format", file=sys.stderr)
        sys.exit(1)
    else:
        print("Configuration loaded")
    return config


def checkConfigurationFormat(config):
    try:
        keras.models.model_from_json(json.dumps(config))
    except ValueError:
        return False
    return True


