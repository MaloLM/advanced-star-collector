from flask import Flask
from .app.routes import RouteConfigurator

app = Flask(__name__)

route_configurator = RouteConfigurator(app)

if __name__ == "__main__":
    app.run(debug=True)
