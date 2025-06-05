import sys
import tempfile
import time

"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_reconstruct.py --model_path output/[path_to_your_model_directory]
python gradio_app_reconstruct_rh20t.py --model_path assets/ur5
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/
python optimize_multiframe.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python optimize_multiframe_batch_grid.py --model_path output/franka_emika_panda4
python optimize_multiframe_batch.py --model_path output/franka_emika_panda_complement1/
with render_feat = False
This file is for UR5
"""


import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2
from torch.optim.lr_scheduler import StepLR
from scipy.spatial.transform import Rotation
from utils_loc.generate_grid_campose import generate_camera_poses

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
from gaussian_renderer import render, render_gradio, render_flow
from scene.cameras import Camera_Pose, Camera
from utils_loc.flow_utils import run_flow_on_images, run_tracker_on_images
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss
from torchvision import transforms

from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T
# from optimize_multiframe import visualize_features
sys.path.insert(0, '/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/dino_vit_features')
# sys.path.append('/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/dino-vit-features')
print("sys.path:", sys.path)
from extractor import ViTExtractor


def flow_to_color(flow):
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0, 1]
    mag = np.sqrt(dx**2 + dy**2)
    mag_max = np.max(mag)
    mag_norm = mag / mag_max if mag_max > 0 else mag
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=2)
    return mcolors.hsv_to_rgb(hsv)

def rotation_matrix_to_quaternion(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    return R.from_quat(quat).as_matrix()

render_feat = False
gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
if render_feat:
    my_feat_decoder = skip_feat_decoder(32).cuda()
    decoder_dict_path = os.path.join(gaussians.model_path, 'point_cloud', 'pose_conditioned_iteration_4000', 'feat_decoder.pth')
    my_feat_decoder.load_state_dict(torch.load(decoder_dict_path))
background_color = torch.zeros((3,)).cuda()
example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'model.xml'))
data = mujoco.MjData(model)
n_joints = model.njnt

def render_scene(c_idx, *args):
    pixels = image_list[c_idx]
    return pixels

def reset_params():
    return [0] * n_joints + [0, -45, 2]

class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*args):
    n_params = len(args)
    n_joints = n_params - 3
    
    joint_angles = torch.tensor(args[:n_joints])
    azimuth, elevation, distance = args[n_joints:]

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
    joint_angles = torch.tensor(joints[Display])
    intrinsic_rh = intrinsics
    fx = intrinsic_rh[0, 0]
    fy = intrinsic_rh[1, 1]
    fovx = 2 * np.arctan(640 / (2 * fx))
    fovy = 2 * np.arctan(360 / (2 * fy))
    camera_extrinsic_matrix = extrinsics
    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                            640, 360, joint_pose=joint_angles, zero_init=True).cuda()
    if FRANKA:
        # frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
        frame = torch.clamp(render_gradio(example_camera_mujoco, gaussians, background_color, pose_normalized=False)['render'], 0, 1)
    elif UR5:
        frame = torch.clamp(render_gradio(example_camera_mujoco, gaussians, background_color, pose_normalized=False)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def gaussian_reset_params():
    return [0] * n_joints + [0, -45, 2]

def random_initialize_joints():
    return [round(np.random.uniform(-1, 1), 2) for _ in range(n_joints)]

def random_initialize_camera():
    random_azimuth = round(np.random.uniform(0, 360), 2)
    random_elevation = round(np.random.uniform(-90, 90), 2)
    random_distance = round(np.random.uniform(1, 3), 2)
    return [random_azimuth, random_elevation, random_distance]

def random_initialize():
    return random_initialize_joints() + random_initialize_camera()

def gaussian_random_initialize_joints():
    return random_initialize_joints()

def gaussian_random_initialize_camera():
    return random_initialize_camera()

def initial_render():
    initial_params = reset_params()
    mujoco_image = render_scene(Display, *initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    return mujoco_image, gaussian_image

def iou_loss(pred, target, smooth=1e-6):
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)  # (batch_size, H*W)
    target = target.view(batch_size, -1)
    intersection = (pred * target).sum(dim=1)  # Sum over pixels per batch
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou).mean()  # Average loss over batch

def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):

    all_cameras = []
    all_joint_poses = []
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Loading DINOv2 model...")
    os.makedirs('inspection_features', exist_ok=True)
    if dino_orig or render_feat:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    mujoco_feats = []
    for c_idx in range(image_list.shape[0]):
        # if c_idx != 3:
        #     continue
        # print('c_idx', c_idx)
        mujoco_image = image_list[c_idx]
        if dino_orig or render_feat:
            dino_image = mujoco_image
            # print(dino_image.shape, 'dino_image')
            dino_image = Image.fromarray(dino_image)
            dino_image.save(f'inspection_features/dino_image_{c_idx}.png')
            dino_image = resize_image(dino_image, 800)
            dino_image = dino_transform(dino_image)[:3].unsqueeze(0)
            dino_image, target_H, target_W = interpolate_to_patch_size(dino_image, dinov2.patch_size)
            dino_image = dino_image.cuda()
            with torch.no_grad():
                features = dinov2.forward_features(dino_image)["x_norm_patchtokens"][0]
            features = features.cpu().numpy()
            # print(features.shape, 'features')
            features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            features_hwc = torch.from_numpy(features_hwc).float().cuda()
            # print(features_hwc.shape, 'features_hwc')
            mujoco_feats.append(features_hwc)
        
        
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1) # TODO permute?
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            # print(mujoco_mask_tensor.shape, 'mujoco_mask_tensor', mujoco_mask_tensor.sum())
            mujoco_masks.append(mujoco_mask_tensor)
        mujoco_depth = depth_list[c_idx]
        mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
        if use_sam_mask:
            mujoco_depth_tensor = mujoco_depth_tensor * mujoco_mask_tensor
        mujoco_depths.append(mujoco_depth_tensor)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(640 / (2 * fx))
        fovy = 2 * np.arctan(360 / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                640, 360, joint_pose=joint_angles, zero_init=True).cuda()
        
            
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
            if Optimal_Camera_Param is not None:
                first_camera.w.data = Optimal_Camera_Param['w'].to(camera.device)
                first_camera.v.data = Optimal_Camera_Param['v'].to(camera.device)
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param

        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)
        print(camera)
        
    print(all_cameras)

    optimizers = []
    schedulers = []
    if optimize_camera:
        optimizer = torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        
    if optimize_joints:
        optimizers.append(torch.optim.Adam(all_joint_poses, lr=joints_lr))

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        for optimizer in optimizers:
            optimizer.zero_grad()

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_feats = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            # if i not in Optimize_list:
            #     print(i)
            #     i += 1
            #     continue
            # camera.joint_pose = joint_pose
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features = render_feat)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features = False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
                    
            elif UR5:
                output = render_gradio(camera, gaussians, background_color, render_features = False)
            
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
                
            if render_feat:
                rendered_feat = output['render_feat'].permute(0, 3, 1, 2)
                dino_feat = my_feat_decoder(rendered_feat).permute(0, 2, 3, 1).squeeze()
                gaussian_feats.append(dino_feat)
            # print(i)
            # print(step)
            
            # if step == 0 and i == 0:
            #     # print(f'-----------------------------------saving pcd from {i}')
            #     os.makedirs("pcd", exist_ok=True)
            #     np.save(f"pcd/means3D_{scene_name}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            # mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            # mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            # os.makedirs('tmp', exist_ok=True)
            # mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        if render_feat:
            gaussian_feats = torch.stack(gaussian_feats)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # mujoco_mask_target = torch.stack(mujoco_masks)
        # mujoco_images_target = torch.stack(mujoco_images)
        # mujoco_depths_target = torch.stack(mujoco_depths)
        
        if tracking_loss and step == 0:
            if use_sam_mask:
                mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            mujoco_depths_target = torch.stack(mujoco_depths)
            print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
            
            fwd_valids = (fwd_valids > 0.3).float()
            
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.1)
            # print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            motion_idx = motion_mask.sum(dim=(1, 2, 3)) > 10000
            motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # print(motion_idx, motion_idx.shape)
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            depth_mask = (mujoco_depths_target[:-1] > 0.5)
            depth_near_mask1 = (mujoco_depths_target < 0.7)
            depth_near_mask = (mujoco_depths_target[:-1] < 0.7)
            depth_near_mask = depth_near_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            motion_depth_near_mask = motion_mask * depth_near_mask
            # print(mujoco_depths_target)
            # print(depth_mask)
            depth_mask = depth_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            
            # print(depth_mask.shape, 'depth mask shape')
            # print(motion_mask)
            # print(motion_mask.shape, 'motion mask shape')
            
            
            
        if tracking_loss and step % 19 == 1 and epoch == 0:
            for i in range(1, min(1, mujoco_images_target.shape[0])):
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                
                flow_2d_np_render = gaussian_flows[i-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                step_grid = 10
                new_image_render = gaussian_tensors[i].permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np_render = flow_to_color(flow_2d_np_render)
                H_f, W_f = flow_2d_np_render.shape[:2]
                x_render, y_render = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx_render = flow_2d_np_render[y_render, x_render, 0]
                dy_render = flow_2d_np_render[y_render, x_render, 1]
                
                flow_2d_mujoco = fwd_flows[i-1].permute(1, 2, 0).detach().cpu().numpy()
                new_image_np = (mujoco_images_target[i] / 255).permute(1, 2, 0).detach().cpu().numpy()
                flow_color_mujoco = flow_to_color(flow_2d_mujoco)
                step_grid = 10
                H_f, W_f = flow_2d_mujoco.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_mujoco[y, x, 0]
                dy = flow_2d_mujoco[y, x, 1]
                
                mask_i = motion_mask[i-1, 0, :, :].float().cpu().numpy()
                depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
                motion_depth_near_mask_i = motion_depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                # print(flow_valid_mask_i.min(), flow_valid_mask_i.max(), 'flow valid mask min max')
                

                axes[0, 0].imshow(new_image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(flow_color_mujoco)
                axes[0, 1].set_title('Flow Color Map')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(new_image_np)
                axes[0, 2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[0, 2].set_title('Image with Flow Arrows')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(new_image_render)
                axes[1, 0].set_title('Rendered Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(flow_color_np_render)
                axes[1, 1].set_title('Rendered Flow Color Map')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(new_image_render)
                axes[1, 2].quiver(x, y, dx_render, dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[1, 2].set_title('Rendered Flow Arrows')
                axes[1, 2].axis('off')

                axes[2, 0].imshow(new_image_np)
                axes[2, 0].imshow(motion_depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 0].set_title('Motion Mask')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(new_image_np)
                axes[2, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 1].set_title('Depth Mask')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(new_image_np)
                axes[2, 2].quiver(x, y, dx - dx_render, dy - dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2, 2].set_title('Image with Flow Arrows')
                axes[2, 2].axis('off')
                
                axes[3, 0].imshow(new_image_np)
                axes[3, 0].imshow(depth_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 0].set_title('Depth Mask')
                axes[3, 0].axis('off')

                axes[3, 1].imshow(new_image_np)
                axes[3, 1].imshow(depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 1].set_title('Valid Mask')
                axes[3, 1].axis('off')

                axes[3, 2].imshow(new_image_np)
                axes[3, 2].imshow(flow_valid_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 2].set_title('Flow valid mask')
                axes[3, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features/super_flow_visualization{i}_{scene_name}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
                
        if dino_orig or render_feat:
            mujoco_feats_target = torch.stack(mujoco_feats)
        
        if render_feat:
            masked_gaussian_feats = gaussian_feats * gaussian_masks.unsqueeze(-1)
            features_final = F.interpolate(masked_gaussian_feats.permute(0, 3, 1, 2), size=mujoco_feats_target.shape[1:3], mode="bilinear", align_corners=False)
            features_final = features_final.permute(0, 2, 3, 1)
            loss_feat = cosine_loss(features_final, mujoco_feats_target)
        
        image = gaussian_tensors
        mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        # print(image.shape, 'image')
        # image_np = (image[0] * 255).permute(1, 2, 0).to(torch.uint8)
        # image_np = image_np.detach().cpu().numpy()
        # image_np = Image.fromarray(image_np)
        # image_np.save(f'inspection_features/images_{0}.png')
        
        # image_np = (mujoco_images_target1[0] * 255).permute(1, 2, 0).to(torch.uint8)
        # image_np = image_np.detach().cpu().numpy()
        # image_np = Image.fromarray(image_np)
        # image_np.save(f'inspection_features/images_mujoco_{0}.png')
        # singularity shell --nv docker://stereolabs/zed:4.2-devel-cuda11.8-ubuntu22.04

        if tracking_loss:
            if motion_idx.sum() <= 5:
                print('valid images < 5')
            #     continue
            # target_mask = (self.input_mask_torch_orig_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float()
            flow_loss = (gaussian_flows - fwd_flows).abs()
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= 640
            flow_loss[:, 1] /= 360
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
        
        if dino_vit:
            image_final = transforms.Resize(100, interpolation=transforms.InterpolationMode.NEAREST)(image)
            mujo_final = transforms.Resize(100, interpolation=transforms.InterpolationMode.NEAREST)(mujoco_images_target1)
            image_final = normalize(image_final)
            mujo_final = normalize(mujoco_images_target1)
            descriptors = extractor.extract_descriptors(image_final.to(device), 11, 'key', False)
            mujo_descriptors = extractor.extract_descriptors(mujo_final.to(device), 11, 'key', False)
            print(descriptors.shape, 'descriptors')
            print(mujo_descriptors.shape, 'mujo_descriptors')
            descriptors = descriptors.squeeze()
            
        
        
        elif dino_orig:
            resized_image = resize_tensor(image, longest_edge=800)
            resized_mujo_image = resize_tensor(mujoco_images_target1, longest_edge=800)
            
            image_final = normalize(resized_image)
            mujo_final = normalize(resized_mujo_image)
            dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
            dino_mojo, _, _ = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
            features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
            mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
            loss_direct_feat = cosine_loss(features, mujo_features)
        
        
        
        threshold = 0.5
        gaussian_masks_binary = (gaussian_masks > threshold).float()  # Convert to 0s and 1s
        if use_sam_mask:
            ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
        gaussian_masks_scaled = (gaussian_masks_binary * 255).byte()  # uint8 format
        if use_sam_mask:
            ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        if use_sam_mask:
            ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 4: Save masks as images
        os.makedirs('masks_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            # Save Gaussian mask
            gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
            # print(f"Saved {gaussian_filename}")

            # Save Ground Truth mask
            if use_sam_mask:
                gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
                cv2.imwrite(gt_filename, ground_truth_masks_np[i])
                # print(f"Saved {gt_filename}")
        
        gaussian_depths_np = gaussian_depths.detach().cpu().numpy()
        mujoco_depths_np = mujoco_depths_target.detach().cpu().numpy()
        gaussian_depths_np *= 40
        mujoco_depths_np *= 40
        os.makedirs('depths_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            # Save Gaussian mask
            gaussian_filename = f"depths_inspection/gaussian_depth_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_depths_np[i])
            # print(f"Saved {gaussian_filename}")

            # Save Ground Truth mask
            gt_filename = f"depths_inspection/ground_truth_depth_{i}.png"
            cv2.imwrite(gt_filename, mujoco_depths_np[i])
            # print(f"Saved {gt_filename}")
        
        # print(gaussian_depths.max(), gaussian_depths.min(), mujoco_depths_target.max(), mujoco_depths_target.min())
        # print(gaussian_depths.shape, 'gaussian_depths')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(mujoco_images_target.shape, 'mujoco_images_target')
        # print(mujoco_mask_target.shape, 'mujoco_mask_target')
        
        # print(mujoco_mask_target.shape, 'mujoco mask shape')
        # print(mujoco_images_target1.shape, 'mujoco image shape')
        # mujoco_images_target1 = mujoco_images_target1 * mujoco_mask_target.unsqueeze(1)
        mse_loss = F.mse_loss(mujoco_images_target1, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3))
        
        if use_sam_mask:
            IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
        # depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
        # depth_diff = depth_loss.mean()
        # if step % 10 == 0 or step == optimization_steps - 1:
        #     print(IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff')
        #     if dino_orig:
        #         print(loss_direct_feat.item(), 'loss_direct_feat')
        #     print(l2_diff.sum(), 'l2')
        
        
        # total_loss = l2_diff.sum() + 3 * IoU_loss + 1 * depth_diff + 1 * loss_feat
        # total_loss = l2_diff.sum() + 1 * IoU_loss
        total_loss = flow_loss
        if total_loss == 0:
            print('Total loss is zero, skipping optimization step.')
            continue
        print(flow_loss, 'flow loss')
        # total_loss = IoU_loss
        # best: 10 depth_diff + 5 IoU_loss + 5 loss_direct_feat
        # total_loss = 5 * depth_diff + 10 * IoU_loss + 10 * loss_feat
        total_loss.backward()

        all_losses = l2_diff.detach().cpu().numpy().tolist()
        all_images = [torch.clamp(tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() for tensor in gaussian_tensors]

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()    

        angle_diff, trans_diff = all_cameras[0].transformation_distance(gt_extrinsics)
        tcp_distance = all_cameras[0].tcp_translation(tcp)
        print('angle_diff', angle_diff, 'trans_diff', trans_diff, 'tcp: ',tcp_distance)
            
            # Extract updated parameters for the best camera
            # updated_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
            # updated_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
            # updated_camera_params = extract_camera_parameters(updated_extrinsic.T)
            
            # updated_params = updated_joint_angles + [
            #     updated_camera_params['azimuth'],
            #     updated_camera_params['elevation'],
            #     updated_camera_params['distance']
            # ]

            # rounded_params = [round(param, 2) for param in updated_params]

            

    print('Camera results robot to world transformation:')
    # for i in range(len(all_cameras)):
    print(all_cameras[0].robot_to_world())
    # print(best_camera.robot_to_world())
    # params = all_cameras[0].get_params()
    angle_diff, trans_diff = all_cameras[0].transformation_distance(gt_extrinsics)
    tcp_distance = all_cameras[0].tcp_translation(tcp, gt_extrinsics)
    print('angle_diff', angle_diff, 'trans_diff', trans_diff, 'tcp: ',tcp_distance.item())
    
    # if 'tcp_distance' in Optimal_Camera_Param:
    #     if tcp_distance < Optimal_Camera_Param['tcp_distance']:
    #         Optimal_Camera_Param['tcp_distance'] = tcp_distance
    #         Optimal_Camera_Param['w'] = all_cameras[0].w
    #         Optimal_Camera_Param['v'] = all_cameras[0].v
    #         print('Optimal_Camera_Param', Optimal_Camera_Param)
    # else:
    #     Optimal_Camera_Param['tcp_distance'] = tcp_distance
    #     Optimal_Camera_Param['w'] = all_cameras[0].w.detach().clone()
    #     Optimal_Camera_Param['v'] = all_cameras[0].v.detach().clone()
        

    # yield (best_image, *rounded_final_params)
    # yield (all_images[Display], *rounded_final_params)
    return [angle_diff, trans_diff, tcp_distance.item()]

W = 640
H = 360

def compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs('inspection_features', exist_ok=True)
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
        # Only perturb parameters that are being optimized
        # print("gaussian_params", gaussian_params)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)


    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features=False)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features=False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
                    
            new_image = output['render']
            new_depth = output['depth']
            # new_pcd = output['pcd']
            
            # if step == 0:
            #     # print(f'-----------------------------------saving pcd from {i}')
            #     os.makedirs("pcd", exist_ok=True)
            #     np.save(f"pcd/means3D_{i}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            os.makedirs('tmp', exist_ok=True)
            mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        # print(gaussian_flows.shape, 'gaussian flows')
        print(step, 'step')
        if tracking_loss and step == 0:
            if use_sam_mask:
                mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            # print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.1)
            # print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            motion_idx = motion_mask.sum(dim=(1, 2, 3)) > 10000
            motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # print(motion_idx, motion_idx.shape)
            # if motion_idx.sum() <= 0:
            #     print('valid images < 5')
            #     torch.cuda.empty_cache()
            #     break
            motion_mask = motion_mask.float() # [T-1, 1, H, W]

        
        if tracking_loss:
            flow_loss = (gaussian_flows - fwd_flows).abs()
            
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids).sum() / fwd_valids.sum()
        
        if True:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/images_{i}.png')
                
                image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/image_render_{i}.png')
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        if use_sam_mask:
            IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
            print(IoU_loss, 'IoU_loss')
            

        flow_loss = flow_loss * 1000
        if flow_loss == 0:
            return None, float('inf')
        
    angle_diff, trans_diff = all_cameras[0].transformation_distance(gt_extrinsics)
    tcp_distance = all_cameras[0].tcp_translation(tcp, gt_extrinsics)
    print('angle_diff', angle_diff, 'trans_diff', trans_diff, 'tcp: ',tcp_distance.item())
    print(flow_loss, 'flow loss')
        

    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), flow_loss.item()
    


def euler_to_extrinsic(pose_6d, euler_order='xyz'):
    """
    Convert a 6D camera pose (translation + Euler angles) to a 4x4 extrinsic matrix.

    Args:
        pose_6d (np.ndarray): 6-element array with [x, y, z, alpha, beta, gamma].
        euler_order (str): Order of Euler angles (e.g., 'xyz', 'zyx'). Default is 'xyz'.

    Returns:
        np.ndarray: 4x4 extrinsic camera matrix.
    """
    # Extract translation vector
    t = pose_6d[:3]

    # Extract Euler angles
    euler_angles = pose_6d[3:]

    # Convert Euler angles to rotation matrix
    R = Rotation.from_euler(euler_order, euler_angles).as_matrix()

    # Initialize 4x4 extrinsic matrix
    extrinsic = np.eye(4)

    # Assign rotation and translation
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    extrinsic = np.linalg.inv(extrinsic)

    return extrinsic


def prepare_flow(images, device, data_file):
    mujoco_images = []
    for c_idx in range(len(images)):
        mujoco_image = images[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
    mujoco_images_target = torch.stack(mujoco_images)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
    # Run cotracker
    fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target) # T-1 2 H W, T-1 H W
    fwd_valids = fwd_valids.unsqueeze(1)
    # Prepare the data by moving tensors to CPU and storing in a dictionary
    data_to_save = {
        'fwd_flows': fwd_flows.detach().cpu(),
        'fwd_valids': fwd_valids.detach().cpu()
    }
    # Save the dictionary to the file
    torch.save(data_to_save, data_file)

if __name__ == "__main__":
    Display = 0
    BATCH_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/rh20t_data_drrobot/'
    FRANKA = True
    UR5 = False
    dino_vit = False
    dino_orig = False
    tracking_loss = True
    use_sam_mask = False
    have_depth = True
    # Optimal_Camera_Param = {'w': torch.tensor([ 0.0267, -0.0028, -0.0195]), 'v': torch.tensor([-0.0042,  0.0215, -0.0006])}
    Optimal_Camera_Param = {'w': torch.tensor([-0.0049, -0.0486, -0.0441]), 'v': torch.tensor([ 0.0121,  0.0143, -0.0107])}
    Optimal_Camera_Param = {'w': torch.tensor([-0.049, -0.0486, -0.0441]), 'v': torch.tensor([ 0.0321,  0.0343, -0.0207])}
    scene_names = os.listdir(BATCH_PATH)
    scene_names = sorted(scene_names)
    print(scene_names)
    

    # extractor = ViTExtractor('dino_vits8', 4, device='cuda')
    all_params = []
    for scene_name in scene_names:
        print(scene_name)
        if scene_name != '036422060215_task_0122_user_0007_scene_0001_cfg_0005':
            continue
        DATA_PATH = os.path.join(BATCH_PATH, scene_name)
        
        image_list_whole = []
        depth_list_whole = []
        mask_list_whole = []
        grippers_whole = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
        joints_whole = np.load(os.path.join(DATA_PATH, 'joints.npy'))
        tcp = np.load(os.path.join(DATA_PATH, 'tcp.npy')) # (3,)
        if UR5:
            fingers_whole = np.stack([grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole], axis=1) # for ur5
        elif FRANKA:
            grippers_whole = grippers_whole * 0
            grippers1_whole = grippers_whole + 0.5
            fingers_whole = np.stack([grippers_whole, grippers_whole], axis=1) # for franka
        if UR5:
            offset_angle = 90 * np.pi / 180
            joints[:, 0] -= offset_angle
        joints_whole = np.concatenate([joints_whole, fingers_whole], axis=1)
        print('joints: ', joints_whole.shape)
        image_names = os.listdir(os.path.join(DATA_PATH, 'images'))
        image_names = sorted(image_names)
        for image_name in image_names:
            # print('image_name', image_name)
            if '.DS_Store' in image_name:
                continue
            image_path = os.path.join(DATA_PATH, 'images', image_name)
            img = Image.open(image_path)
            img = np.array(img)
            image_list_whole.append(img)
        image_list_whole = np.stack(image_list_whole)
            
        depth_names = os.listdir(os.path.join(DATA_PATH, 'depths'))
        depth_names = sorted(depth_names)
        for depth_name in depth_names:
            # print('depth_name', depth_name)
            if '.DS_Store' in depth_name:
                continue
            depth_path = os.path.join(DATA_PATH, 'depths', depth_name)
            depth = np.load(depth_path)
            depth_list_whole.append(depth)
        depth_list_whole = np.stack(depth_list_whole)
        
        if use_sam_mask:
            mask_names = os.listdir(os.path.join(DATA_PATH, 'masks'))
            mask_names = sorted(mask_names)
            for mask_name in mask_names:
                # print('mask_name', mask_name)
                if '.DS_Store' in mask_name:
                    continue
                mask_path = os.path.join(DATA_PATH, 'masks', mask_name)
                mask = Image.open(mask_path)
                mask = np.array(mask)
                mask_list_whole.append(mask)
            mask_list_whole = np.stack(mask_list_whole)
        # print(mask_list.shape, 'masks ensure not 3 channels')
        assert(depth_list_whole.shape[0] == image_list_whole.shape[0] == grippers_whole.shape[0] == joints_whole.shape[0])
        # print('frames: ', image_list.shape[0])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        data_file = os.path.join(DATA_PATH, 'tracker_data.pth')
        if not os.path.exists(data_file):
            prepare_flow(image_list_whole, device, data_file)
        data_loaded = torch.load(data_file)
        fwd_flows_whole = data_loaded['fwd_flows']
        fwd_valids_whole = data_loaded['fwd_valids']
        
        intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
        gt_extrinsics = np.load(os.path.join(DATA_PATH, 'extrinsics.npy'))
        print(gt_extrinsics)
        
        cam_poses = generate_camera_poses('left', num_samples=100)
        best_pose = None
        best_score = float('inf')
        epoch = 0
        first_camera = None  # Track the first camera to share its parameters
        print('cam_poses: ', cam_poses[0])
        time.sleep(1)
        for pose in cam_poses:
            extrinsics = euler_to_extrinsic(pose)
            # extrinsics = gt_extrinsics
            optimize_camera = False
            optimize_joints = False
            camera_lr = 0.0
            joints_lr = 0.0
            optimization_steps = 1
            Optimize_list = [j for j in range(30)]
            image_list = image_list_whole[Optimize_list]
            joints = joints_whole[Optimize_list]
            grippers = grippers_whole[Optimize_list]
            fingers = fingers_whole[Optimize_list]
            fwd_flows = fwd_flows_whole[Optimize_list[:-1]]
            fwd_valids = fwd_valids_whole[Optimize_list[:-1]]
            params, flow_loss_item = compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        fwd_flows, fwd_valids)
            if flow_loss_item < best_score:
                best_score = flow_loss_item
                best_pose = pose
        print('best pose: ', best_pose)
        print(best_score, 'best score')
        # pose = np.array([0, 0.57382269, 0.84167107, -2.02259688, -0.08240233,
        #                  -2.40839443])
        pose = np.array([0.006111708272572871, 0.7799910768896609, 0.48825174062541965, -1.5580715300648718, 0.0, -2.5771203117437884])
        # extrinsics = euler_to_extrinsic(pose)
        # wait a second
        time.sleep(10)
        extrinsics = euler_to_extrinsic(pose)
        print(extrinsics)
            
        torch.cuda.empty_cache()
        first_camera = None  # Track the first camera to share its parameters

        l = len(image_list_whole)
        batch_size = 20
        steps_per_epoch = l // batch_size
        
        epochs = 20
        
        for epoch in tqdm(range(epochs)):
            
            for batch_idx in range(steps_per_epoch):
                Optimize_list = []
                for image_idx in range(batch_size):
                    if image_idx + batch_idx * batch_size >= l:
                        break
                    Optimize_list.append(batch_idx * batch_size + image_idx)
                print(Optimize_list)
                if len(Optimize_list) < batch_size - 10:
                        continue
                image_list = image_list_whole[Optimize_list]
                depth_list = depth_list_whole[Optimize_list]
                if use_sam_mask:
                    mask_list = mask_list_whole[Optimize_list]
                joints = joints_whole[Optimize_list]
                grippers = grippers_whole[Optimize_list]
                fingers = fingers_whole[Optimize_list]
                fwd_flows = fwd_flows_whole[Optimize_list[:-1]]
                fwd_valids = fwd_valids_whole[Optimize_list[:-1]]
                print(image_list.shape, 'images')
        
                optimize_camera = True
                optimize_joints = False
                camera_lr = 0.0005
                camera_lr = camera_lr * (0.5 ** (epoch // 4))
                joints_lr = 0.001
                optimization_steps = 10
                powerful_optimize_dropdown = "Disabled"
                noise_input = 0.01
                num_inits_input = 1
                W = 640
                H = 360
                
                params = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                                fwd_flows, fwd_valids)
        all_params.append(params)
    last_elements = [param[-1] for param in all_params]
    print('final results')
    print(last_elements)
    result_path = os.path.join(DATA_PATH, 'last_elements.txt')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as file:
        for element in last_elements:
            file.write(str(element) + '\n')
    print(all_params[:, -1])