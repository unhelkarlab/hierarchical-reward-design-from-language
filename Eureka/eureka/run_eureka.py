import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed_idx", nargs='+', type=int, default=0)
args = parser.parse_args()
seed_indices = args.seed_idx

for index in seed_indices:
  print(f"Running with seed_idx={index}...")
  command = [
      "python", "eureka.py", "env=thor_pnp_flatsa", "sample=1", "iteration=1",
      "model=gpt-4o", f"seed_idx={index}", "blocking=True", "num_options=4"
  ]

  # Run the command and wait until it's done
  result = subprocess.run(command)

  # Optional: check if it failed
  if result.returncode != 0:
    print(
        f"Command failed at index {index} with return code {result.returncode}")

# for index in range(0, 1):
#   print(f"Running with seed_idx={index}...")
#   command = [
#       "python", "eureka.py", "env=thor_pnp_ll", "sample=1", "iteration=1",
#       "model=gpt-4o", f"seed_idx={index}", "blocking=True", "num_options=4"
#   ]

#   # Run the command and wait until it's done
#   result = subprocess.run(command)

#   # Optional: check if it failed
#   if result.returncode != 0:
#     print(
#         f"Command failed at index {index} with return code {result.returncode}")
