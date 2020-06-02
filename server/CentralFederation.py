import sys
import threading
import time

SLEEP_TIME = 1
UPDATES_PER_ROUND = 3
ROUNDS = 3

# TODO Proper logging


class Federation:
    def __init__(self, model):
        self.t1 = threading.Thread(target=self.run)
        # TODO Ensure thread safety for the following variables
        self.running = False
        self.model = model
        self.updates = []
        self.round = 0

    def start(self):
        self.running = True
        self.t1.start()

    def run(self):
        while self.running:
            time.sleep(SLEEP_TIME)

            if len(self.updates) >= UPDATES_PER_ROUND:
                print(str(len(self.updates)), "/", str(UPDATES_PER_ROUND), " - Averaging")

                self.model.aggregateUpdates(self.updates)
                self.updates = []

                evaluation = self.model.evaluate()
                print(evaluation)

                self.round = self.round + 1

                if self.round >= ROUNDS:
                    self.running = False
                    self.model.writeToDisk()
                    print("Finished training, you can safely shutdown the process now...")

            else:
                print(str(len(self.updates)), "/", str(UPDATES_PER_ROUND), " - Waiting for more trainings")

    def stop(self):
        self.running = False
        self.t1.join()

    def join(self):
        self.t1.join()

    def getGlobalModel(self):
        return self.model.serialize()

    def submitUpdates(self, client_id, weights):
        # TODO is one client allowed to submit multiple updates per round?
        # TODO Reject updates from older rounds
        # TODO plausibility check of incoming data
        # TODO Don't accept updates while averaging
        self.updates.append([client_id, weights])
