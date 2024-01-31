from flask import Flask
from .app.routes import RouteConfigurator
from .agent.agent_manager import DQNAgentManager
from .agent.model_updater_thread import ModelUpdaterThread

app = Flask(__name__)
agent_manager = DQNAgentManager()
model_updater = ModelUpdaterThread(agent_manager)
route_configurator = RouteConfigurator(
    app, agent_manager, model_updater)

if __name__ == "__main__":
    app.run(debug=True)
