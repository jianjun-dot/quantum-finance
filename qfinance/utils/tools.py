import inspect
import json
import os
import torch

def results_to_JSON(result):
    result_JSON = {}
    for name, value in inspect.getmembers(result):
        if (
            not name.startswith("_")
            and not inspect.ismethod(value)
            and not inspect.isfunction(value)
            and hasattr(result, name)
        ):

            result_JSON[name] = value
    return result_JSON

def save_meta_data(strike_name: str, date: str, time:str, meta_data={}):
    cwd = os.getcwd()
    new_directory = cwd + "/data/"+ date + "/" + strike_name + "/"
    try:
        os.makedirs(new_directory, exist_ok=True)
        print(f"Directory '{new_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{new_directory}' already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    meta_file_name = new_directory + strike_name + "_" + date +"-"+ time + "_meta.json"
    with open(meta_file_name, "w") as f:
        json.dump(meta_data, f)

def save_JSON(strike_name: str, date: str, time:str, data: dict):
    cwd = os.getcwd()
    new_directory = cwd+ "/data/"+ date + "/" + strike_name + "/"
    try:
        os.makedirs(new_directory, exist_ok=True)
        print(f"Directory '{new_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{new_directory}' already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    file_name=  new_directory + strike_name + "_" + date +"-"+ time + ".json"
    with open(file_name, "w") as f:
        json.dump(data, f)
        
def time_convert(sec):
    if sec > 3600:
        hours = sec // 3600
        sec = sec % 3600
        minutes = sec // 60
        sec = sec % 60
        return "{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),sec)
    elif sec > 60:
        minutes = sec // 60
        sec = sec % 60
        return "{:0>2}m:{:05.2f}s".format(int(minutes),sec)
    else:
        return "{:05.2f}s".format(sec)
    
def check_gpu_availability():
    return torch.cuda.is_available()