BATCH_PATH="/data/group_data/katefgroup/datasets/robomind/robomind_chenyu/auto_calibration/241021_close_trash_bin_1"

SCENE_NAMES=($(ls -d "$BATCH_PATH"/*/ | xargs -n 1 basename | sort))

START_INDEX=0

eval "$(conda shell.bash hook)"

i=0
for scene_name in "${SCENE_NAMES[@]}"; do
    # Construct full scene path
    scene_path="$BATCH_PATH/$scene_name"
    if [ "$i" -ge "$START_INDEX" ]; then
        echo "Running Python script for Index: $i, Scene Path: $scene_path"
        
        conda deactivate
        conda activate dr
        cd /data/user_data/wenhsuac/chenyuzhang/backup/drrobot

        python robomind_calibration/autocalibration_singletraj.py --model_path output/franka_fr3_2f85_highres_finetune_0 --scene_path "$scene_path"
        python robomind_calibration/render_singletraj.py --model_path output/franka_fr3_2f85_complement_1 --scene_path "$scene_path"
    fi
    ((i++))
done

# bash robomind_calibration/run_batch.sh 