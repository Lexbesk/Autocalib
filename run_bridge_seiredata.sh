
BATCH_PATH="/data/group_data/katefgroup/datasets/bridge_chenyu/yidi/chenyu/bridge_seriedata"  # Replace with your actual path
# BATCH_PATH="/data/group_data/katefgroup/datasets/bridge_chenyu/yidi/chenyu/bridge_apr29"
# BATCH_PATH="/data/group_data/katefgroup/datasets/bridge_chenyu/yidi/chenyu/bridge_ori"

# Step 1: List directories in BATCH_PATH
SCENE_NAMES=($(ls -d "$BATCH_PATH"/*/ | xargs -n 1 basename))

declare -A name_to_num
for name in "${SCENE_NAMES[@]}"; do
    # Extract numeric part (assuming prefix like 'scene', e.g., scene123 -> 123)
    num=$(echo "$name" | sed 's/^scene//')
    # Store name with numeric part as key (ensuring it's treated as integer)
    name_to_num[$num]=$name
done

# Sort numeric keys as integers and reconstruct sorted SCENE_NAMES
SCENE_NAMES=()
for num in $(printf "%s\n" "${!name_to_num[@]}" | sort -n); do
    SCENE_NAMES+=("${name_to_num[$num]}")
done

START_INDEX=12

# Step 2: Sort directories by the numeric part (assuming names like scene123)
# SCENE_NAMES=($(printf "%s\n" "${SCENE_NAMES[@]}" | sort -t 'e' -k 2 -n))
eval "$(conda shell.bash hook)"
# Step 3: Iterate with index and pass --scene_path to Python script
i=0
for scene_name in "${SCENE_NAMES[@]}"; do
    # Construct full scene path
    scene_path="$BATCH_PATH/$scene_name"
    if [ "$i" -ge "$START_INDEX" ]; then
        echo "Running Python script for Index: $i, Scene Path: $scene_path"
        
        # cd /data/user_data/wenhsuac/chenyuzhang/moge_video
        # # Activate the conda environment
        # conda deactivate
        # conda activate mega_sam
        # python moge_static_bridge.py --scene_path "$scene_path"

        conda deactivate
        conda activate dr
        cd /data/user_data/wenhsuac/chenyuzhang/backup/drrobot

        # Call Python script with --scene_path argument
        python optimize_multiframe_bridge_single_seriedata.py --model_path output/widow0 --scene_path "$scene_path"
    fi
    ((i++))
done