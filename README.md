<div>

# Auto-Calibration

### ----

To run calibration on Droid, Robomind and Bridge datasets.


## Setup 🛠️

Run the following to create an environment

```
conda create -n dr_test python=3.10
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
gsplat==1.4.0

pip install tensorboard ray tqdm mujoco open3d plyfile pytorch-kinematics random-fourier-features-pytorch pytz gradio

pip install moviepy==1.0.3
pip install opencv-python
pip install timm
pip install transformations
```

The most tricky dependency of our codebase is [gsplat](https://github.com/nerfstudio-project/gsplat), which is used for rasterizing Gaussians. We recommend visiting their installation instructions if the plain `pip install` doesn't work. 
 
# Running Calibration on Datasets 😎



Use the following scripts for calibration on batch of data, or use the python files inside them to run on a single video. Note that this code requires specific data structure, which can be checked in the data paths provided in the script.
```bash
# for robomind
bash run_robomind_seiredata.sh

# for droid
bash run_droid_seiredata.sh

# for bridge
bash run_bridge_seiredata.sh
```

Important: Modify the robot model checkpoint path inside the scripts to the following path: 
```bash
# for robomind and droid (franka fr3)
--model_path /data/group_data/katefgroup/datasets/chenyu/drrobot/output/franka_fr3_2f85_highres_finetune_0

# for bridge (widow)
--model_path /data/group_data/katefgroup/datasets/chenyu/drrobot/output/widow0
```

Then run the following scripts to visualize the results. Please change the data path in the command.
```bash
# for robomind
python render_batch_robomind_single_seriedata.py --model_path /data/group_data/katefgroup/datasets/chenyu/drrobot/output/franka_fr3_2f85_highres_finetune_0 --scene_path /data/group_data/katefgroup/datasets/robomind/robomind_chenyu/robomind_extract_1/scene_0

# for droid
python render_droid_seriedata.py --model_path /data/group_data/katefgroup/datasets/chenyu/drrobot/output/franka_fr3_2f85_complement_1 --scene_path /data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_3/scene_27

# for bridge
python render_batch_bridge_single.py --model_path /data/group_data/katefgroup/datasets/chenyu/drrobot/output/widow0 --scene_path /data/group_data/katefgroup/datasets/bridge_chenyu/yidi/chenyu/bridge_seriedata/scene16
```
