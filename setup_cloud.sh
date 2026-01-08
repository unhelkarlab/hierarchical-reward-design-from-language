sudo apt install libvulkan1 vulkan-tools
sudo apt install libegl1
git clone --branch ai2thor --single-branch https://github.com/unhelkarlab/hierarchical_reward_design.git
cd /workspace/hierarchical_reward_design/
conda env create -f conda_env.yaml
conda activate hrd
pip install -e .
pip install -e Eureka
pip install -e Hierarchical-Language-Agent/agent
pip install -e Hierarchical-Language-Agent/testbed-cooking
wandb login
cd HierRL

conda activate hrd ; cd /workspace/hierarchical_reward_design/HierRL ; python train/run_ll.py --env_name pnp --pref_type task --model_type PPO --seed_idx 0 --option_to_use 0