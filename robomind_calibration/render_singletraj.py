import sys
import tempfile
from moviepy.editor import VideoFileClip
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)



os.environ['MUJOCO_GL'] = 'egl'
import cv2
import pickle
from torch.optim.lr_scheduler import StepLR

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir

import gradio as gr

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

import numpy as np
import mujoco
from utils_loc.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from tqdm import tqdm

from video_api import initialize_gaussians
from gaussian_renderer import render, render_gradio
from scene.cameras import Camera_Pose, Camera
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss
from torchvision import transforms

from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T

from robomind_calibration.data import INIT_W2C


@ torch.no_grad()
def render(folder_path):
    all_cameras = []
    mujoco_images = []
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1) # TODO permute?
        mujoco_images.append(mujoco_tensor)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        
        all_cameras.append(camera)


    gaussian_tensors = []
    gaussian_masks = []
    gaussian_depths = []
    gaussian_feats = []
    i = 0
    for camera in tqdm(all_cameras):
        output = render_gradio(camera, gaussians, background_color, render_features=False)
        new_image = output['render']
        gaussian_tensors.append(new_image)
        i += 1
        
    
    gaussian_tensors = torch.stack(gaussian_tensors)
    mujoco_images_target = torch.stack(mujoco_images)
    
    
    image = gaussian_tensors
    mujoco_images_target1 = mujoco_images_target / 255
    
    for ii in range(image.shape[0]):
        image_np_render = (image[ii] * 255).permute(1, 2, 0).to(torch.uint8)
        image_np = image_np_render.detach().cpu().numpy()
        image_np = Image.fromarray(image_np)
        
        image_np_mujo = (mujoco_images_target1[ii] * 255).permute(1, 2, 0).to(torch.uint8)
        image_np = image_np_render * 0.5 + image_np_mujo * 0.5
        image_np = image_np.to(torch.uint8)
        image_np = image_np.detach().cpu().numpy()
        image_np = Image.fromarray(image_np)
        image_np.save(os.path.join(folder_path, f'{ii}.png'))
    
    return

def render_with_video(folder_path, output_video_path):
    os.makedirs(folder_path, exist_ok=True)

    render(folder_path)

    images = [img for img in os.listdir(folder_path) if img.endswith((".jpg", ".png"))]
    images.sort(key=lambda x: int(x.split('.')[0])) 

    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30 FPS

    # Add each image to the video
    for image in images:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write frame to video

    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")
    clip = VideoFileClip(output_video)
    gif_filename = os.path.splitext(output_video)[0] + '.gif'
    gif_path = os.path.join(DATA_PATH, gif_filename)
    clip.write_gif(gif_path, fps=10)
    clip.close()


"""
How to use:

python robomind_calibration/render_singletraj.py --model_path output/franka_fr3_2f85_complement_1 --scene_path /data/group_data/katefgroup/datasets/robomind/robomind_chenyu/auto_calibration/2024_09_20_pick_fruit_and_bread/0920_155013

"""

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
    background_color = torch.zeros((3,)).cuda()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene directory')
    parser.add_argument('--model_path', type=str, default='output/franka_fr3_2f85_highres_finetune_0', help='Path to the scene directory')
    args = parser.parse_args()
    FRANKA = True
    UR5 = False
    dino_vit = False
    dino_orig = False
    shift_joints = True
    Optimal_Camera_Param = None
    ROBOTIQ = True
    
    TRAJ_PATH = args.scene_path
    BATCH_PATH = os.path.dirname(TRAJ_PATH)
    print('TRAJ_PATH: ', TRAJ_PATH)
    print('BATCH_PATH: ', BATCH_PATH)
    results_file = os.path.join(BATCH_PATH, 'results.pkl')
    with open(results_file, 'rb') as f:
        results_dict = pickle.load(f)
    all_params = []
    camera_names = ['camera_left', 'camera_right']
    for camera_name in camera_names:
        DATA_PATH = os.path.join(TRAJ_PATH, camera_name)
        extrinsics = INIT_W2C[camera_name]
        image_list = []
        depth_list = []
        mask_list = []
        grippers = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
        joints = np.load(os.path.join(DATA_PATH, 'joints.npy'))
        fingers = grippers * np.array([[0.75, -0.4, 0.6, -0.26, 0.75, -0.4, 0.6, -0.26]])
        joints = np.concatenate([joints, fingers], axis=1)
        # if shift_joints:
        #     # shift joints back k frame
        #     joints = np.concatenate([joints[-4:], joints[4:]], axis=0)
        image_names = os.listdir(os.path.join(DATA_PATH, 'images'))
        image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
        for image_name in image_names:
            # print('image_name', image_name)
            if '.DS_Store' in image_name:
                continue
            image_path = os.path.join(DATA_PATH, 'images', image_name)
            img = Image.open(image_path)
            img = np.array(img)
            image_list.append(img)
        image_list = np.stack(image_list)
        _, H, W, C = image_list.shape
        
        image_list = image_list[::10]  # Downsample by 4
        joints = joints[::10]
        

        intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
        # extrinsics = np.load(os.path.join(DATA_PATH, 'extrinsics.npy'))
        keys = list(results_dict.keys())
        
        extrinsics = np.array(results_dict[keys[0]][camera_name]['extrinsics'])
        print('extrinsics', extrinsics)

        folder_path = os.path.join(DATA_PATH, 'blends')  # Replace with your folder path
        output_video = os.path.join(DATA_PATH, f'video_blend.mp4')    # Output video file name
        render_with_video(folder_path, output_video)
        
        extrinsics = INIT_W2C[camera_name]
        folder_path = os.path.join(DATA_PATH, 'blends_init')
        output_video = os.path.join(DATA_PATH, f'video_blend_init.mp4')    # Output video file name
        render_with_video(folder_path, output_video)
