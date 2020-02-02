import tensorflow as tf
import json
import numpy as np

# TODO Implement modelling through subclassing, make KerasModel abstract


class KerasModel:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(28, 28)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(10, activation='softmax')
            ])
        self.model.compile(optimizer='adam',
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


