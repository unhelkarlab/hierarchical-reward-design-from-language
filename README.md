# Hierarchical Reward Design from Language

This repository contains an implementation corresponding to the paper "Hierarchical Reward Design from Language: Enhancing Alignment of Agent Behavior with Human Specifications", to appear in AAMAS 2026.

### Setup

After cloning the repository and changing the current working directory (using `cd`), install the required dependencies in a Conda environment by running the following the commands:

```
conda env create -f conda_env.yaml
conda activate hrdl

pip install -e .
pip install -e Eureka
pip install -e Hierarchical-Language-Agent/agent
pip install -e Hierarchical-Language-Agent/testbed-cooking
```

Our LLM-based reward generation uses the OpenAI API. In order to run the related experiments, set your API key using the command below:

```
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

### Training

To replicate the experiments shown in Table 1 of the paper, we split up hierarchical policy training into three categories: LLM-generated hierarchical preference rewards (our framework),  LLM-generated flat preference rewards (baseline), and task-only reward (baseline).

**Using LLM-Generated Hierarchical Preference Rewards**

To train a low-level policy:

`TODO`

To train a high-level policy:

`TODO`

**Using LLM-Generated Flat Preference Rewards**

To train a full hierarchical policy:

`TODO`

**Using Task-Only Rewards**

To train a low-level policy:

(run from the `HierRL` folder)
```
python train/run_ll.py \
    --env_name [ENV] \
    --pref_type task \
    --model_type PPO \
    --seed_idx 0 \
    --option_to_use [OPTION]
```

where `[ENV]` is one of `[rw4t, oc, pnp]` (corresponding to the Rescue World, Kitchen, and iThor domains respectively) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type task \
    --model_type VariableStepDQN \
    --seed_idx 0 \
```

where `[ENV]` is defined similar to above. Make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy. 

---

To replicate the proof-of-concept experiments shown in Table 3 in the Appendix of the paper, we split up hierarchical policy training into two categories: expert-defined hierarchical rewards and expert-defined flat rewards (the task-only reward results in Table 3 are the same as in Table 1).

**Using Expert-Defined Hierarchical Rewards**

To train a low-level policy:

(run from the `HierRL` folder)
```
python train/run_ll.py \
    --env_name [ENV] \
    --pref_type low \
    --model_type PPO \
    --seed_idx 0 \
    --option_to_use [OPTION]
```

where `[ENV]` is one of `[rw4t, oc, pnp]` (corresponding to the Rescue World, Kitchen, and iThor domains respectively) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type high \
    --model_type VariableStepDQN \
    --seed_idx 0
```

where `[ENV]` is defined similar to above. Make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy. 

**Using Expert-Defined Flat Rewards**

To train a low-level policy:

(run from the `HierRL` folder)
```
python train/run_ll.py \
    --env_name [ENV] \
    --pref_type flatsa \
    --model_type PPO \
    --seed_idx 0 \
    --option_to_use [OPTION]
```

where `[ENV]` is one of `[rw4t, oc, pnp]` (corresponding to the Rescue World, Kitchen, and iThor domains respectively) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type flatsa \
    --model_type VariableStepDQN \
    --seed_idx 0
```

where `[ENV]` is defined similar to above. Make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy.

### Evaluating Policies

TODO