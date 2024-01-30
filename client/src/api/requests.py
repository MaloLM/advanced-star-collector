import requests
import json

API_URL = "http://127.0.0.1:5000"


def hello_world():
    url = f"{API_URL}/helloworld"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Something went wrong while calling the API")


def get_action(state, mode):
    url = f"{API_URL}/get_action"
    data = {"state": state, "mode": mode}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["action"]
    else:
        raise Exception("Failed to get action from server")


def update_model(training_data):
    url = f"{API_URL}/update_model"
    response = requests.post(url, json=training_data)
    if response.status_code != 200:
        raise Exception("Failed to update model on server")


def end_training(filename):
    url = f"{API_URL}/end_training"
    data = {"filename": filename}
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception("Failed to end training on server")
