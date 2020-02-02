from flask import Flask, Response, request
from CentralFederation import Federation
from Model import KerasModel
import json

app = Flask(__name__)
fed = Federation(KerasModel())


@app.route("/model", methods=["GET"])
def hello_world():
    return Response(fed.getGlobalModel(), mimetype="application/json")


@app.route("/update", methods=["POST"])
def update():
    data = json.loads(request.json)
    fed.submitUpdates(data["client_id"], data["weights"])
    return "Update submitted successfully."


if __name__ == '__main__':
    fed.start()
    app.run()
