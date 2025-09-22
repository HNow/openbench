import json
import os
from inspect_ai.dataset import MemoryDataset, Sample

def record_to_sample(record: dict,metadata: dict = {}) -> Sample:
    return Sample(
        input=record["instruction"],
        target='',
        metadata={
            "instruction": record["instruction"],
            "actions": record["actions"],
            "outputs": record["outputs"],
            **metadata
        },
    )




# gets the actual tasks, ie from the train splits
# TODO: update this to respect the split
def get_taubench_dataset(env_type: str = "retail", task_split: str = "test",metadata: dict = {}):
    # Get the directory path for the task data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", env_type)
    
    # Load the task file based on split
    task_file = f"tasks_{task_split}.json"
    with open(os.path.join(data_dir, task_file)) as f:
        tasks = json.load(f)
    
    # Convert to Samples
    samples = [record_to_sample(task,metadata) for task in tasks]
    # only 1 sample
    samples = samples[:1]
    return MemoryDataset(samples=samples, name=f"taubench_{env_type}_{task_split}")

# gets the task task specific data for that environment
# this is the data over which the agent operates


def get_task_env_data(env_type: str = "retail"):
    if env_type == "retail":
        return get_retail_env_data()
    elif env_type == "airline":
        return get_airline_env_data()
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

def get_retail_env_data():
    # Get the directory path for the environment data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "retail")
    
    with open(os.path.join(data_dir, "orders.json")) as f:
        order_data = json.load(f)
    with open(os.path.join(data_dir, "products.json")) as f:
        product_data = json.load(f)
    with open(os.path.join(data_dir, "users.json")) as f:
        user_data = json.load(f)
    return {
        "orders": order_data,
        "products": product_data,
        "users": user_data,
    }

def get_airline_env_data():
    # Get the directory path for the environment data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "airline")
    
    with open(os.path.join(data_dir, "flights.json")) as f:
        flight_data = json.load(f)
    with open(os.path.join(data_dir, "bookings.json")) as f:
        booking_data = json.load(f)
    with open(os.path.join(data_dir, "passengers.json")) as f:
        passenger_data = json.load(f)
    return {
        "flights": flight_data,
        "bookings": booking_data,
        "passengers": passenger_data,
    }