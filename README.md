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

To replicate the experiments shown in Table 1 of the paper, we split up hierarchical policy training into two categories: LLM-generated hierarchical rewards (our framework) and LLM-generated flat rewards (baseline).

**Using LLM-Generated Hierarchical Rewards**

To train a low-level policy:

`TODO`

To train a high-level policy:

`TODO`

**Using LLM-Generated Flat Rewards**

To train a low-level policy:

`TODO`

To train a high-level policy:

`TODO`

To replicate the proof-of-concept experiments shown in Table 3 in the Appendix of the paper, we split up hierarchical policy training into two categories: expert-defined hierarchical rewards and expert-defined flat rewards.

**Using Expert-Defined Hierarchical Rewards**

To train a low-level policy:

`TODO`

To train a high-level policy:

`TODO`

**Using Expert-Defined Flat Rewards**

To train a low-level policy:

`TODO`

To train a high-level policy:

`TODO`

### Evaluating Policies

TODO