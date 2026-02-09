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

Our LLM-based reward generation framework uses the OpenAI API. In order to run the related experiments, set your API key using the command below:

```
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

### Training

To replicate the experiments shown in Table 1 of the paper, we split up hierarchical policy training into three categories: LLM-generated hierarchical preference rewards (our framework),  LLM-generated flat preference rewards (baseline), and task-only reward (baseline).

**Using LLM-Generated Preference Rewards**

To train a policy using the LLM-based reward generation pipeline:

(run from the `Eureka/eureka` folder)
```
python run_eureka.py \
    --exp_type [POLICY_LEVEL] \
    --env [ENV] \
    --seed_idx 0
```

To train a low-level policy with low-level generated rewards:
- Set `--exp_type` to `low_level`
- Set `--env` to either `rw4t` (Rescue World) or `pnp` (iTHOR)

To train a high-level policy with high-level generated rewards:
- Set `--exp_type` to `high_level`
- Set `--env` to either `rw4t` (Rescue World), `oc` (Kitchen), or `pnp` (iTHOR)

To train a hierarchical policy with flat state-action generated rewards:
- Set `--exp_type` to `flat`
- Set `--env` to either `rw4t` (Rescue World), `oc` (Kitchen), or `pnp` (iTHOR)
- **NOTE**: If `--env` is `pnp`, this command will only train the low-level policy. To train the high-level policy, run `python train/run_hl_flatsa_eureka_parallel_nocheck.py --env_name pnp --seed_idx 0 --model_type VariableStepDQN` from the `HierRL` folder (you will not need to run postprocessing as described below for this)

**IMPORTANT**: After training a low-level policy, you will need to postprocess the training runs to filter out for generated rewards that are syntactically correct before training a high-level policy. You can do so with the following command (run from the `HierRL` folder)

```
python eval/eval_ll.py \
    --env_name [ENV] \
    --class_name [CLASS_NAME] \
    --pref_type low \
    --model_type PPO \
    --seed_idx 0
```

If you are postprocessing Rescue World policies, set `--env` to `rw4t` and `--class_name` to `RescueWorldLLGPT`

If you are postprocessing iTHOR policies, set `--env` to `pnp` and `--class_name` to `ThorPickPlaceEnvLLGPT`

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

where `[ENV]` is one of `[rw4t, pnp]` (corresponding to the Rescue World and iThor domains respectively, note that Kitchen is limited to high-level training) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type task \
    --model_type VariableStepDQN \
    --seed_idx 0 \
```

where `[ENV]` is defined similar to above. For the Rescue World and iThor environments, make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy. 

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

where `[ENV]` is one of `[rw4t, pnp]` (corresponding to the Rescue World and iThor domains respectively, note that Kitchen is limited to high-level training) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type high \
    --model_type VariableStepDQN \
    --seed_idx 0
```

where `[ENV]` is defined similar to above. For the Rescue World and iThor environments, make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy. 

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

where `[ENV]` is one of `[rw4t, pnp]` (corresponding to the Rescue World and iThor domains respectively, note that Kitchen is limited to high-level training) and `option_to_use` indicates the specific option to train (you will need to train a policy for each of the options separately).

To train a high-level policy:

(run from the `HierRL` folder)
```
python train/run_hl.py \
    --env_name [ENV] \
    --pref_type flatsa \
    --model_type [MODEL_TYPE] \
    --seed_idx 0
```

where `[ENV]` is defined similar to above, and `[MODEL_TYPE]` is either `VariableStepDQN` for Rescue World and iTHOR, or `MaskableDQN` for Kitchen. For the Rescue World and iThor environments, make sure that `--seed_idx` matches an index that was used to train a set of low-level policies, as those policies will be used to train the high-level policy.

### Evaluating Policies

To evaluate trained policies from LLM-generated rewards:

(run from the `HierRL` folder)
```
python eval/eval_hl.py \
    --env_name [ENV] \
    --pref_type [PREF] \
    --model_type [MODEL TYPE] \
    --class_name [CLASS NAME]
```

where `--pref_type` can be chosen from `[high, flatsa]` depending on if you want to evaluate models trained with hierarchical or flat preference rewards.

If you are evaluating on the Rescue World environment:
- Set `--env_name` to `rw4t`
- Set `--model_type` to `VariableStepDQN`
- Set `--class_name` to `ThorPickPlaceEnvHLGPT` if `--pref_type` is `high`, and `ThorPickPlaceEnvFlatSAGPT` if `--pref_type` is `flatsa`

If you are evaluating on the Kitchen environment:
- Set `--env_name` to `oc`
- Set `--model_type` to `MaskableDQN`
- Set `--class_name` to `KitchenHLGPT` if `--pref_type` is `high`, and `KitchenFlatSAGPT` if `--pref_type` is `flatsa`

If you are evaluating on the iTHOR environment:
- Set `--env_name` to `pnp`
- Set `--model_type` to `VariableStepDQN`
- Set `--class_name` to `ThorPickPlaceEnvHLGPT` if `--pref_type` is `high`, and `ThorPickPlaceEnvFlatSAGPT` if `--pref_type` is `flatsa`

To evaluate trained policies using either task-only reward or expert-provided flat/hierarchical rewards:

(run from the `HierRL` folder)
```
python eval/eval_hl.py \
    --env_name [ENV] \
    --pref_type [PREF] \
    --model_type [MODEL TYPE]
```

where `--pref_type` can be chosen from `[task, high, flatsa]` depending on if you want to evaluate models trained with task-only, hierarchical preference rewards, or flat preference rewards.

If you are evaluating on the Rescue World environment:
- Set `--env_name` to `rw4t`
- Set `--model_type` to `VariableStepDQN`

If you are evaluating on the Kitchen environment:
- Set `--env_name` to `oc`
- Set `--model_type` to `MaskableDQN`

If you are evaluating on the iTHOR environment:
- Set `--env_name` to `pnp`
- Set `--model_type` to `VariableStepDQN`