import json
import requests

API_URL = "http://127.0.0.1:5000"


def hello_world():
    url = f"{API_URL}/helloworld"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Something went wrong while calling the API")


def get_action(state, mode, epsilon):
    url = f"{API_URL}/get_action"
    data = {"state": state, "mode": mode, "epsilon": epsilon}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["action"]
    else:
        raise Exception("Failed to get action from server")


def serialize_experience(experience):
    state, action, reward, next_state, done = experience[0]
    total_reward = experience[1]
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "total_reward": total_reward
    }


def update_model(training_data):
    url = f"{API_URL}/update_model"
    # preparing data
    serialized_experiences = []

    for exp in training_data.buffer:
        serialized_experiences.append(serialize_experience(exp))

    response = requests.post(url, json=serialized_experiences)
    if response.status_code != 200:
        raise Exception("Failed to update model on server")


def start_training(modelname: str):
    url = f"{API_URL}/start_training"
    data = {"modelname": modelname}
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception("Failed to end training on server")


def end_training():
    url = f"{API_URL}/end_training"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to end training on server")


def get_queue_size():
    url = f"{API_URL}/queue_size"
    response = requests.get(url)
    if response.status_code == 200:
        response_data = response.json()
        queue_size = response_data.get("queue_size")
        return queue_size
    else:
        print(
            f"Failed to retrieve queue size, status code: {response.status_code}")
