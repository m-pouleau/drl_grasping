import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TASK = "GraspPlanetary"
ENV = "Octree"
WINDOW_SIZE = 1

def moving_average(data, window_size):
    """Apply centered moving average filter to data with padding."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to center the window.")
    
    pad_width = window_size // 2
    padded_data = np.concatenate([np.repeat(data.to_numpy()[0], pad_width), data.to_numpy()])
    
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(padded_data, weights, 'valid')

# Initialize a list to store dataframes
dataframes = []

# Load CSV files into a pandas DataFrame
file_pattern = f"{ENV}_Seed_*_tensorboard_training_data.csv"
file_paths = glob.glob(os.path.join(f'/root/drl_grasping_training/training_data/{TASK}/', file_pattern))
for file in file_paths:
    # Read each CSV file into a dataframe
    df = pd.read_csv(file)
    # Append the dataframe to the list
    dataframes.append(df[['Step', 'eval/mean_reward', 'eval/success_rate']])

# Concatenate all dataframes on 'Step' to align them
combined_df = pd.concat(dataframes).reset_index(drop=True)

# Group by 'Step' and calculate the mean for each step
averaged_df = combined_df.groupby('Step').mean().reset_index()

# Define the output file path and save new dataframe to a CSV file
output_file_path = os.path.join(f'/root/drl_grasping_training/training_data/{TASK}/', f"{ENV}_merged_tensorboard_training_data.csv")
averaged_df.to_csv(output_file_path, index=False)

# Get smoothed values for merged runs (WINDOW_SIZE = 1 -> no smoothing)
smoothed_mean_reward = moving_average(averaged_df['eval/mean_reward'], WINDOW_SIZE)
smoothed_success_rate = moving_average(averaged_df['eval/success_rate'], WINDOW_SIZE)
if WINDOW_SIZE != 1:
    smoothed_steps = averaged_df['Step'].iloc[:int(-(WINDOW_SIZE-1)/2)]
else:
    smoothed_steps = averaged_df['Step']

# Plot 1: 'eval/mean_reward' vs 'Step'
plt.figure(figsize=(10, 6))
plt.plot(smoothed_steps, smoothed_mean_reward, linestyle='-', color='b', label='Mean Reward')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.title('Mean Episode Reward during Evaluation')
plt.grid(True)
plt.legend()
if WINDOW_SIZE == 1:
    plt.savefig(f'/root/drl_grasping_training/training_data/{TASK}/{ENV}_merged_mean_reward_no_smoothing.png')
else:
    plt.savefig(f'/root/drl_grasping_training/training_data/{TASK}/{ENV}_merged_mean_reward_0{WINDOW_SIZE}_smoothing.png')
plt.close()

# Plot 2: 'eval/success_rate' vs 'Step'
plt.figure(figsize=(10, 6))
plt.plot(smoothed_steps, smoothed_success_rate, linestyle='-', color='g', label='Success Rate')
plt.xlabel('Step')
plt.ylabel('Success Rate')
plt.title('Mean Success Rate during Evaluation')
plt.grid(True)
plt.legend()
if WINDOW_SIZE == 1:
    plt.savefig(f'/root/drl_grasping_training/training_data/{TASK}/{ENV}_merged_success_rate_no_smoothing.png')
else:
    plt.savefig(f'/root/drl_grasping_training/training_data/{TASK}/{ENV}_merged_success_rate_0{WINDOW_SIZE}_smoothing.png')
plt.close()

print("Plots saved as mean_reward_plot.png and success_rate_plot.png")
