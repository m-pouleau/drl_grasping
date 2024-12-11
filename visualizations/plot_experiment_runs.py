import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EXP_LIST = [
    # "PoinNet Feature Extractor (with Colors) - No Pre-Training",
    # "PointNet - Pretrained Classification - XYZ channels",
    # "PointNet - Pretrained Sem Segmentation - XYZRGB channels",
    # "O-CNN Feature Extractor (baseline)",
    # "O-CNN - Dynamic Lift Threshold - Constant Timestep Penalty",
    "O-CNN - Incremental Lift Reward - Constant Timestep Penalty",
    "O-CNN - Incremental Lift Reward - Exponentially Growing Timestep Penalty",
    "O-CNN - Incremental Lift Reward - Linearly Growing Timestep Penalty",
    ]
SMOOTHING_VALUES = [1, 1, 1]

def moving_average(data, window_size):
    """Apply centered moving average filter to data with padding."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to center the window.")
    
    pad_width = window_size // 2
    padded_data = np.concatenate([np.repeat(data.to_numpy()[0], pad_width), data.to_numpy()])
    
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(padded_data, weights, 'valid')

SMOOTHED_STEPS = []
SMOOTHED_MEAN_REWARD = []
SMOOTHED_SUCCESS_RATE = []

for i in range(len(EXP_LIST)):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(f'{EXP_LIST[i]}.csv')

    SMOOTHED_MEAN_REWARD.append(moving_average(df['eval/mean_reward'], SMOOTHING_VALUES[i]))
    SMOOTHED_SUCCESS_RATE.append(moving_average(df['eval/success_rate'], SMOOTHING_VALUES[i]))
    if SMOOTHING_VALUES[i] != 1:
        SMOOTHED_STEPS.append(df['Step'].iloc[:int(-(SMOOTHING_VALUES[i]-1)/2)])
    else:
        SMOOTHED_STEPS.append(df['Step'])

# Plot 1: 'eval/mean_reward' vs 'Step'
plt.figure(figsize=(10, 6))
for i in range(len(EXP_LIST)):
    plt.plot(SMOOTHED_STEPS[i], SMOOTHED_MEAN_REWARD[i], linestyle='-', label=f'{EXP_LIST[i]}')
plt.xlabel('Step', fontsize=14)
plt.ylabel('Mean Reward', fontsize=14)
plt.title('Learning Curve of trained agents - Evalution of Mean Reward', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig(f'Comparison Learning Curve - Mean Reward.png')
plt.close()

# Plot 2: 'eval/success_rate' vs 'Step'
plt.figure(figsize=(10, 6))
for i in range(len(EXP_LIST)):
    plt.plot(SMOOTHED_STEPS[i], SMOOTHED_SUCCESS_RATE[i], linestyle='-', label=f'{EXP_LIST[i]}')
plt.xlabel('Step', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.title('Learning Curve of trained agents - Evalution of Success Rate', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig(f'Comparison Learning Curve - Success Rate.png')
plt.close()

print("Plots saved as mean_reward_plot.png and success_rate_plot.png")
