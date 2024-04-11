import subprocess
import psutil
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, NVMLError
from datetime import datetime

# Initialize NVML
nvmlInit()

# Get handle for the first GPU
handle = nvmlDeviceGetHandleByIndex(0)

# Path to your script
script_path = "../scripts/vanilla_call.py"

# Get the current date and time
current_datetime = datetime.now()

# Convert to string using strftime
date_time_string = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")

# # run on CPU first
# use_GPU = "False"
# # Start your script as a subprocess
# process = subprocess.Popen(["python", script_path, use_GPU], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # Create a psutil object for monitoring
# p = psutil.Process(process.pid)
# # Open the file for writing



# with open("memory_tracker_CPU-{}.txt".format(date_time_string), "w") as file:
#     try:
#         start_time = time.time()
#         while process.poll() is None:  # Check if the subprocess is still running
#             # CPU Memory Usage
#             mem_info = psutil.Process(process.pid).memory_info()
#             cpu_usage = f"CPU Memory Usage - RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB\n"
            
#             # GPU Memory Usage
#             gpu_mem_info = nvmlDeviceGetMemoryInfo(handle)
#             gpu_usage = f"GPU Memory Used: {gpu_mem_info.used / (1024 ** 2):.2f} MB\n\n"
            
#             # Write to file
#             file.write(cpu_usage)
#             # file.write(gpu_usage)
#             file.flush()  # Flush after each write to make sure data is written to disk
            
#             time.sleep(1)  # Adjust based on how frequently you want to check
#         end_time = time.time()
#         file.write(f"Total time: {end_time - start_time} s")
#     except psutil.NoSuchProcess:
#         file.write("The process has finished.\n")
#     except NVMLError as error:
#         file.write(f"Failed to access NVIDIA GPU. Error: {str(error)}\n")

# # run on CPU first
use_GPU = "True"
# Start your script as a subprocess
process = subprocess.Popen(["python", script_path, use_GPU], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # Create a psutil object for monitoring
p = psutil.Process(process.pid)
# Open the file for writing

with open("memory_tracker_GPU-{}.txt".format(date_time_string), "w") as file:
    try:
        start_time = time.time()
        while process.poll() is None:  # Check if the subprocess is still running
            # CPU Memory Usage
            mem_info = psutil.Process(process.pid).memory_info()
            cpu_usage = f"CPU Memory Usage - RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB\n"
            
            # GPU Memory Usage
            gpu_mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_usage = f"GPU Memory Used: {gpu_mem_info.used / (1024 ** 2):.2f} MB\n\n"
            
            # Write to file
            file.write(cpu_usage)
            # file.write(gpu_usage)
            file.flush()  # Flush after each write to make sure data is written to disk
            
            time.sleep(0.5)  # Adjust based on how frequently you want to check
        end_time = time.time()
        file.write(f"Total time: {end_time - start_time} s")
    except psutil.NoSuchProcess:
        file.write("The process has finished.\n")
    except NVMLError as error:
        file.write(f"Failed to access NVIDIA GPU. Error: {str(error)}\n")


# Shut down NVML
nvmlShutdown()

# Capture and print subprocess output and errors
stdout, stderr = process.communicate()
print("STDOUT:", stdout.decode())
print("STDERR:", stderr.decode())

