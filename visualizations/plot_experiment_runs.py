import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EXP_LIST = [
    # "Octree with Intensity - OCNN",
    # "PointNet with Intensity - Training Plain Network",
    # "Octree without Color - OCNN",
    # "PointNet without Color - Using Pretrained Classification Network",
    "Octree with Color - OCNN",
    "PointNet with Color - Using Pretrained Segmentation Network",
    "PointNet with Color - Using Pretrained Segmentation Network with Pointwise Features",
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
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.title('Learning Curve of trained agents - Evalution of Mean Reward')
plt.grid(True)
plt.legend()
plt.savefig(f'Comparison Learning Curve - Mean Reward.png')
plt.close()

# Plot 2: 'eval/success_rate' vs 'Step'
plt.figure(figsize=(10, 6))
for i in range(len(EXP_LIST)):
    plt.plot(SMOOTHED_STEPS[i], SMOOTHED_SUCCESS_RATE[i], linestyle='-', label=f'{EXP_LIST[i]}')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.title('Learning Curve of trained agents - Evalution of Success Rate')
plt.grid(True)
plt.legend()
plt.savefig(f'Comparison Learning Curve - Success Rate.png')
plt.close()

print("Plots saved as mean_reward_plot.png and success_rate_plot.png")
