#!/bin/bash

# Maximum number of concurrent tmux sessions
MAX_SESSIONS=12

# Parse command-line arguments
train=false
env_name="rw4t"
pref_type="task"
start_idx=0
num_seeds=1
render=false
model_type="DQN"

# "$#" refers to the number of remaining command-line arguments
while [[ "$#" -gt 0 ]]; do
    # "$1" represents the current argument, and "$2" (if applicable) is its 
    # value.
    # $@ contains all remaining arguments.
    case $1 in
        --train) train=true ;;
        --env_name) env_name="$2"; shift ;;
        --pref_type) pref_type="$2"; shift ;;
        --start_idx) start_idx="$2"; shift ;;
        --num_seeds) num_seeds="$2"; shift ;;
        --render) render=true ;;
        --model_type) model_type="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to wait until there are less than MAX_SESSIONS tmux sessions running
wait_for_free_slot() {
    while true; do
        running_sessions=$(tmux ls 2>/dev/null | wc -l)
        if [ "$running_sessions" -lt "$MAX_SESSIONS" ]; then
            break
        fi
        sleep 5
    done
}

# Determine which Python script to run based on train/eval mode
case "$train" in
    true)  python_script="train/run_hl.py" ;;  # Training script
    false) python_script="eval/eval_hl.py" ;;   # Evaluation script
    *) echo "Invalid value for --train: $train"; exit 1 ;;
esac


# Variables
end_idx=$((start_idx+num_seeds))
train_str=$([[ "$train" == true ]] && echo "train" || echo "eval")
# Keep tmux open if doing eval
keep_tmux_open=$([[ "$train" == false ]] && echo "exec bash" || echo "")

# Loop through seeds
for idx in $(seq $start_idx $((end_idx - 1))); do

    # Wait for an available tmux slot
    wait_for_free_slot

    # Create a unique tmux session name
    session_name="${env_name}_HL_${model_type}_seed-idx${idx}_${train_str}_${pref_type}"

    echo "Running experiment in tmux session: ${session_name}"

    # Run the experiment in a new detached tmux session
    tmux new-session -d -s ${session_name} "bash -c 'source ~/anaconda3/bin/activate hrd && \
        python ${python_script} \
            --env_name ${env_name} \
            --pref_type ${pref_type} \
            --seed_idx ${idx} \
            --model_type ${model_type} \
            $( [[ ${render} == true ]] && echo '--render' ); ${keep_tmux_open} '"
done
