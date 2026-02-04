import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--exp_type", required=True, choices=["flat", "low_level", "high_level"])
parser.add_argument("--env", required=True, choices=["rw4t", "oc", "pnp"])
parser.add_argument("--seed_idx", nargs='+', type=int, default=0)
args = parser.parse_args()
seed_indices = args.seed_idx

env_types = {
  'rw4t': {
    'flat': 'rescue_world_flatsa',
    'low_level': 'rescue_world_ll',
    'high_level': 'rescue_world_hl',
  },
  'oc': {
    'flat': 'kitchen_flatsa',
    'high_level': 'kitchen_hl',
  },
  'pnp': {
    'flat': 'thor_pnp_flatsa',
    'low_level': 'thor_pnp_ll',
    'high_level': 'thor_pnp_hl',
  }
}

run_args = {
  'env': env_types[args.env][args.exp_type],
  'sample': 1,          # How many rewards to generate
  'num_options': -1,     # How many options to train per generated reward
  'blocking': False,     # If False, train policies in parallel (if sample > 1)
}

if args.env == "pnp" and args.exp_type != "high":
  run_args['blocking'] = True
  run_args['num_options'] = 4

for index in seed_indices:
  print(f"Running with seed_idx={index}...")
  command = [
      "python", "eureka.py", f"env={run_args['env']}", f"sample={run_args['sample']}", "iteration=1",
      "model=gpt-4o", f"seed_idx={index}", f"blocking={run_args['blocking']}", f"num_options={run_args['num_options']}"
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
