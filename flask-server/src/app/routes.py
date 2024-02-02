import logging
import sys
from flask import request, jsonify, Flask
from ..logger.logging import setup_loggers
from ..agent.agent_manager import DQNAgentManager
from ..utils.game_states import RANDOM, TESTING, TRAINING
from ..agent.model_updater_thread import ModelUpdaterThread


setup_loggers()
logger = logging.getLogger('app_logger')


class RouteConfigurator:
    def __init__(self, app: Flask, agent_manager: DQNAgentManager, model_updater: ModelUpdaterThread):
        self.app: Flask = app
        self.agent_manager = agent_manager
        self.thread = model_updater
        self.configure_routes()
        self.thread.check_and_restart_thread()

    def configure_routes(self):

        @self.app.route('/start_training', methods=['POST'])
        def start_training():
            modelname = request.json.get('modelname')
            self.thread.check_and_restart_thread()
            logger.info(f"Starting training with {modelname}")
            self.agent_manager.reset_agent(modelname)
            self.thread.tf_logger.step_count = 0
            return jsonify({"message": "Started"}), 200

        @self.app.route('/end_training', methods=['GET'])
        def end_training():
            self.thread.stop()
            logger.info(
                f"Ending training with model: {self.agent_manager.agent.modelname}")
            return jsonify({"message": "Model saved"}), 200

        @self.app.route('/get_action', methods=['POST'])
        def get_action():
            logger.info("Action requested")
            data = request.json

            state = data['state']
            mode = data['mode']
            epsilon = data['epsilon']
            modelname = data['modelname']

            agent = self.agent_manager.agent

            if mode == TRAINING:
                action = agent.choose_action_for_training(
                    state, epsilon)
            elif mode == TESTING:
                action = agent.choose_action_with_model(
                    state, modelname)
            elif mode == RANDOM:
                action = agent.choose_random_action()
            else:
                return jsonify({"error": "Invalid mode"}), 400

            return jsonify({"action": action}), 200

        @self.app.route('/update_model', methods=['POST'])
        def update_model():
            experiences_data = request.json

            experiences = []
            episode_failed = True
            for exp in experiences_data:
                experiences.append((exp["state"], exp["action"], exp["reward"],
                                   exp["next_state"], exp["done"], exp["total_reward"]))

                if (exp['done'] == True and exp['next_state'][0][3] == 1.0) or exp['state'][1] == 1:
                    episode_failed = False
                    break
                elif exp['done'] == True and exp['next_state'][0][0] == 1.0:
                    episode_failed = True
                    break

            self.agent_manager.update_experience_replay(
                experiences, episode_failed)

            return jsonify({"message": "Data received and queued for processing"}), 200

        @self.app.route('/queue_size', methods=['GET'])
        def get_queue_size():
            queue_size = self.agent_manager.update_queue.qsize()
            return jsonify({"queue_size": queue_size}), 200

        @self.app.route('/is_model_saved', methods=['POST'])
        def is_model_saved():
            data = request.json

            modelname = data['modelname']
            is_model_saved = self.agent_manager.agent.is_model_saved(modelname)
            return jsonify({"is_model_saved": is_model_saved}), 200
