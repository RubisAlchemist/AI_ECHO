import subprocess
import time

# Command to run
command = ["python", "inference/real3d_infer.py", "--src_img", "mina.jpg", "--drv_aud", "audio.wav", "--out_mode", "final", "--out_name", "timeinference.mp4"]

# List to store execution times
execution_times = []

# Run the command 100 times
for i in range(100):
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    
    # Print each execution time
    print(f"Run {i+1}: {execution_time:.2f} seconds")

# Compute the average execution time
average_time = sum(execution_times) / len(execution_times)
print(f"\nAverage execution time over 100 runs: {average_time:.2f} seconds")
