from flask import Flask, request, jsonify
from agent.dqn_agent import DQNAgent
from threading import Thread
import queue

agent = DQNAgent()
update_queue = queue.Queue()


def configure_routes(app):

    def update_model_from_queue():
        while True:
            if not update_queue.empty():
                data = update_queue.get()
                print(data)
                agent.buffer.add(data)
                agent.check_buffer_to_update_policy()

    model_updater_thread = Thread(target=update_model_from_queue)
    model_updater_thread.start()

    @app.route('/helloworld', methods=['GET'])
    def hello_world():
        data = {"state": "test", "mode": 2}
        return data

    @app.route('/end_training', methods=['POST'])
    def end_training():
        filename = request.json.get('filename')
        agent.save_model(filename)
        return jsonify({"message": "Model saved"}), 200

    @app.route('/get_action', methods=['POST'])
    def get_action():
        data = request.json
        state = data['state']
        mode = data['mode']

        if mode == "TRAINING":
            action = agent.choose_action_for_training(state)
        elif mode == "TESTING":
            action = agent.choose_action_with_model(state)
        elif mode == "RANDOM":
            action = agent.choose_random_action()
        else:
            return jsonify({"error": "Invalid mode"}), 400

        return jsonify({"action": action}), 200

    @app.route('/update_model', methods=['POST'])
    def update_model():
        data = request.json
        update_queue.put(data)
        return jsonify({"message": "Data received and queued for processing"}), 200
