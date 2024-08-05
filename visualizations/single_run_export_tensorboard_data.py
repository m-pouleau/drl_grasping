import csv
from tensorboard.backend.event_processing import event_accumulator

# Set environment
ENV = "GraspPlanetary-PointNetWithIntensity-Gazebo-v0"
ALGO = "TQC_3"
FILE = "events.out.tfevents.zzzzzzzzz.ubuntu.xxx.y" # "events.out.tfevents.1720050307.ubuntu.1740.0"
FILE_INDEX = "abc"

# Path to the directory containing the TensorBoard logs
log_dir = f'/root/drl_grasping_training/train/{ENV}/tensorboard_logs/{ENV}/{ALGO}/'

event_file_path = log_dir + FILE

# Scalars to extract
scalars = ['eval/mean_reward', 'eval/success_rate']

# Initialize EventAccumulator to load event files
ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

# Prepare CSV file
with open(f'/root/drl_grasping_training/train/{ENV}/{ALGO}_tensorboard_training_data_{FILE_INDEX}.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Step'] + scalars)  # Write header
    # add zero row at beginning
    csvwriter.writerow([0, 0, 0])

    # Extract and write scalar data
    steps = ea.Scalars(scalars[0])
    for step in steps:
        row = [step.step]
        for scalar in scalars:
            scalar_data = ea.Scalars(scalar)
            scalar_value = next((s.value for s in scalar_data if s.step == step.step), None)
            row.append(scalar_value)
        csvwriter.writerow(row)

print(f"Data extracted to tensorboard_training_data_{FILE_INDEX}.csv")
