import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")

EXP_LIST = [
    "O-CNN - Octree-Based Feature Extractor",
    "PointNet - Pretrained Sem. Seg. - Approach 2 With Skip Connection",
    "3D-Diffusion-Policy-Based Feature Extractor",
    ]
SMOOTHING_VALUES = [3, 3, 3]

def moving_average(data, window_size):
    """Apply centered moving average filter to data with padding."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to center the window.")
    
    pad_width = window_size // 2
    padded_data = np.concatenate([np.repeat(data.to_numpy()[0], pad_width), data.to_numpy()])
    
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(padded_data, weights, 'valid')

# Create empty DataFrame for Seaborn
plot_data = pd.DataFrame(columns=["Step", "Mean Reward", "Success Rate", "Experiment"])

for i, exp_name in enumerate(EXP_LIST):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(f'{EXP_LIST[i]}.csv')

    mean_reward_smooth = moving_average(df['eval/mean_reward'], SMOOTHING_VALUES[i])
    success_rate_smooth = moving_average(df['eval/success_rate'], SMOOTHING_VALUES[i])

    if SMOOTHING_VALUES[i] != 1:
        steps = df['Step'].iloc[:int(-(SMOOTHING_VALUES[i]-1)/2)]
    else:
        steps = df['Step']

    # Append to plot_data DataFrame
    temp_df = pd.DataFrame({
        "Step": steps,
        "Mean Reward": mean_reward_smooth,
        "Success Rate": success_rate_smooth,
        "Experiment": exp_name
    })

    plot_data = pd.concat([plot_data, temp_df], ignore_index=True)


sns.set(rc={'figure.figsize':(10, 6)})

# Plot 1: 'eval/mean_reward' vs 'Step'
ax1 = sns.lineplot(data=plot_data, x="Step", y="Mean Reward", hue="Experiment", linewidth=2.5)
ax1.set_title("Learning Curve of trained agents - Evaluation of Mean Reward", fontsize=18)
ax1.set_xlabel("Step", fontsize=16)
ax1.set_ylabel("Mean Reward", fontsize=16)
#y_min, y_max = ax1.get_ylim()
#ax1.set_ylim(y_min, y_max+23)
ax1.legend(title=None, fontsize=12, loc='best') # loc can be 'lower right', 'upper left', 'lower center', ...
ax1.get_figure().savefig("01_Comparison Learning Curve - Mean Reward.png")
ax1.get_figure().clf()  # Clear figure

# Plot 2: 'Success Rate' vs 'Step'
ax2 = sns.lineplot(data=plot_data, x="Step", y="Success Rate", hue="Experiment", linewidth=2.5)
ax2.set_title("Learning Curve of trained agents - Evaluation of Success Rate", fontsize=18)
ax2.set_xlabel("Step", fontsize=16)
ax2.set_ylabel("Success Rate", fontsize=16)
#y_min, y_max = ax2.get_ylim()
#ax2.set_ylim(y_min, y_max+0.022)
ax2.legend(title=None, fontsize=12, loc='best') # loc can be 'lower right', 'upper left', 'lower center', ...
ax2.get_figure().savefig("02_Comparison Learning Curve - Success Rate.png")
ax2.get_figure().clf()  # Clear figure


print("Plots saved as mean_reward_plot.png and success_rate_plot.png")