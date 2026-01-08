import subprocess

for index in range(0, 5):
  print(f"Running with seed_idx={index}...")
  command = [
      "python", "train/run_flatsa.py", "--env_name", "rw4t", "--pref_type",
      "flatsa", "--seed_idx",
      str(index), "--model_type", "VariableStepDQN"
  ]

  # Run the command and wait until it's done
  result = subprocess.run(command)

  # Optional: check if it failed
  if result.returncode != 0:
    print(
        f"Command failed at index {index} with return code {result.returncode}")
