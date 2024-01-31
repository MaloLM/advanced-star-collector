import sys
import logging
import queue
from threading import Thread
from ..agent.dqn_agent import DQNAgent
from flask import request, jsonify
from ..logging.logging import setup_loggers

setup_loggers()
app_logger = logging.getLogger('app_logger')


class RouteConfigurator:
    def __init__(self, app):
        self.agent = DQNAgent()
        self.update_queue = queue.Queue()
        self.app = app
        self.configure_routes()
        self.start_model_updater()

    def start_model_updater(self):
        def update_model_from_queue():
            while True:
                if not self.update_queue.empty():
                    data = self.update_queue.get()
                    print(data)
                    self.agent.buffer.add(data)
                    self.agent.check_buffer_to_update_policy()

        model_updater_thread = Thread(target=update_model_from_queue)
        model_updater_thread.start()

    def configure_routes(self):
        @self.app.route('/helloworld', methods=['GET'])
        def hello_world():
            data = {"state": "test", "mode": 2}
            return data

        @self.app.route('/get_epsilon', methods=['GET'])
        def get_epsilon():
            epsilon = self.agent.epsilon

            print(f'EPSI {epsilon}', file=sys.stdout)

            return jsonify({"epsilon": epsilon}), 200

        @self.app.route('/start_training', methods=['POST'])
        def start_training():
            filename = request.json.get('filename')
            # INSTANCIER CORRECTEMEN,T
            # agent.(filename)
            return jsonify({"message": "Started"}), 200

        @self.app.route('/end_training', methods=['POST'])
        def end_training():
            filename = request.json.get('filename')
            agent.save_model(filename)
            return jsonify({"message": "Model saved"}), 200

        @self.app.route('/get_action', methods=['POST'])
        def get_action():
            data = request.json
            state = data['state']
            mode = data['mode']

            if mode == "TRAINING":
                action = self.agent.choose_action_for_training(state)
            elif mode == "TESTING":
                action = self.agent.choose_action_with_model(state)
            elif mode == "RANDOM":
                action = self.agent.choose_random_action()
            else:
                return jsonify({"error": "Invalid mode"}), 400

            return jsonify({"action": action}), 200

        @self.app.route('/update_model', methods=['POST'])
        def update_model():
            data = request.json
            self.update_queue.put(data)
            return jsonify({"message": "Data received and queued for processing"}), 200
