import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ENV = "GraspPlanetary-PointNetWithColor-Gazebo-v0"
ALGO = "TQC_1"
WINDOW_SIZE = 1

def moving_average(data, window_size):
    """Apply centered moving average filter to data with padding."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to center the window.")
    
    pad_width = window_size // 2
    padded_data = np.concatenate([np.repeat(data.to_numpy()[0], pad_width), data.to_numpy()])
    
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(padded_data, weights, 'valid')

# Load CSV file into a pandas DataFrame
df = pd.read_csv(f'/root/drl_grasping_training/train/{ENV}/{ALGO}_tensorboard_training_data.csv')

smoothed_mean_reward = moving_average(df['eval/mean_reward'], WINDOW_SIZE)
smoothed_success_rate = moving_average(df['eval/success_rate'], WINDOW_SIZE)
if WINDOW_SIZE != 1:
    smoothed_steps = df['Step'].iloc[:int(-(WINDOW_SIZE-1)/2)]
else:
    smoothed_steps = df['Step']

# Plot 1: 'eval/mean_reward' vs 'Step'
plt.figure(figsize=(10, 6))
plt.plot(smoothed_steps, smoothed_mean_reward, linestyle='-', color='b', label='Mean Reward')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.title('Mean Episode Reward during Evaluation')
plt.grid(True)
plt.legend()
plt.savefig(f'/root/drl_grasping_training/train/{ENV}/{ALGO}_mean_reward_plot.png')
plt.close()

# Plot 2: 'eval/success_rate' vs 'Step'
plt.figure(figsize=(10, 6))
plt.plot(smoothed_steps, smoothed_success_rate, linestyle='-', color='g', label='Success Rate')
plt.xlabel('Step')
plt.ylabel('Success Rate')
plt.title('Mean Success Rate during Evaluation')
plt.grid(True)
plt.legend()
plt.savefig(f'/root/drl_grasping_training/train/{ENV}/{ALGO}_success_rate_plot.png')
plt.close()

print("Plots saved as mean_reward_plot.png and success_rate_plot.png")
