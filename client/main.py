import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
import json
import uuid
import random

url = "http://localhost:5000"
client_id = uuid.uuid4()
PARTITIONS = 3


def getData(trainer_id):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = [x for x in np.split(x_train, PARTITIONS)][trainer_id]
    y_train = [y for y in np.split(y_train, PARTITIONS)][trainer_id]

    x_train = x_train / 255.0

    return x_train, y_train


def run(trainer_id):

    config = None
    weights = None

    try:
        response = requests.get(url+"/model")
        print("Status: ", response.status_code, response.headers)

        if response.status_code == 200:
            content = response.json()
            config = content['config']
            weights = content['weights']
            weights = [np.array(w) for w in weights]

    except Exception as error:
        print(error)
        return

    if config is None or weights is None:
        print("Error 1")
        return

    model = keras.models.model_from_json(json.dumps(config))
    model.set_weights(weights)

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train, y_train = getData(trainer_id)

    model.fit(x_train, y_train, epochs=1)

    weights = model.get_weights()
    weights = [w.tolist() for w in weights]

    data = {'weights': weights, 'client_id': str(client_id)}

    try:
        response = requests.post(url + '/update', json=json.dumps(data))
        print("Status: ", response.status_code, response.headers)

        if response.status_code == 200:
            print("Successfully send model updates")

    except Exception as error:
        print(error)
        return


if __name__ == '__main__':
    for i in range(0, PARTITIONS):
        run(i)

