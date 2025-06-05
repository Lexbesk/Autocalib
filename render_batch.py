import sys
import tempfile

"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_reconstruct.py --model_path output/[path_to_your_model_directory]
python gradio_app_reconstruct_rh20t.py --model_path assets/ur5
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/
python optimize_multiframe.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python render_batch.py --model_path output/franka_emika_panda4
python optimize_multiframe_batch.py --model_path output/franka_emika_panda_complement1/
with render_feat = False
This file is for UR5
"""


import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2
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
# from optimize_multiframe import visualize_features
sys.path.insert(0, '/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/dino_vit_features')
# sys.path.append('/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/dino-vit-features')
print("sys.path:", sys.path)
from extractor import ViTExtractor



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

def copy_mujoco_camera(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[n_joints:]

def copy_mujoco_joints(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[:n_joints]

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

def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, powerful_optimize_dropdown, noise_input, num_inits_input):

    if powerful_optimize_dropdown == "Enabled":
        num_inits = int(num_inits_input)
        grid_size = int(np.sqrt(num_inits))
    else:
        num_inits = 1
        grid_size = 1
        noise_input = 0

    all_cameras = []
    all_joint_poses = []
    
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

    for init in range(num_inits):
        # mujoco_masks = []
        mujoco_images = []
        # mujoco_depths = []
        # mujoco_feats = []
        first_camera = None  # Track the first camera to share its parameters
        for c_idx in range(image_list1.shape[0]):
            # if c_idx != 3:
            #     continue
            if c_idx not in Optimize_list:
                continue
            # print('c_idx', c_idx)
            mujoco_image = image_list1[c_idx]
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
            # mujoco_mask = mask_list[c_idx]
            # mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            # mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            # print(mujoco_mask_tensor.shape, 'mujoco_mask_tensor', mujoco_mask_tensor.sum())
            # mujoco_masks.append(mujoco_mask_tensor)
            # mujoco_depth = depth_list[c_idx]
            # mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
            # mujoco_depth_tensor = mujoco_depth_tensor * mujoco_mask_tensor
            # mujoco_depths.append(mujoco_depth_tensor)
            
            joint_angles = torch.tensor(joints[c_idx+t])
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

    optimizers = []
    schedulers = []
    if optimize_camera:
        optimizer = torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
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
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            # if i not in Optimize_list:
            #     print(i)
            #     i += 1
            #     continue
            camera.joint_pose = joint_pose
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features = render_feat)
            elif UR5:
                output = render_gradio(camera, gaussians, background_color, render_features = False)
            
            new_image = output['render']
            # new_depth = output['depth']
            # new_pcd = output['pcd']
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
            # new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            # gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            # thresholded = torch.sigmoid((new_image - 0.01) * 500)
            # mask_tensor = thresholded.max(dim=0)[0]
            # mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            # mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            # os.makedirs('tmp', exist_ok=True)
            # mask_image.save('tmp/extracted_mask.png')
            # gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        # gaussian_masks = torch.stack(gaussian_masks)
        # gaussian_depths = torch.stack(gaussian_depths)
        if render_feat:
            gaussian_feats = torch.stack(gaussian_feats)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # mujoco_mask_target = torch.stack(mujoco_masks)
        mujoco_images_target = torch.stack(mujoco_images)
        # mujoco_depths_target = torch.stack(mujoco_depths)
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
        
        print(image.shape, 'image')
        print(mujoco_images_target1.shape)
        os.makedirs(os.path.join(DATA_PATH, 'renders'), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, 'blends'), exist_ok=True)
        for ii in range(image.shape[0]):
            image_np_render = (image[ii] * 255).permute(1, 2, 0).to(torch.uint8)
            image_np = image_np_render.detach().cpu().numpy()
            # print(image_np.shape)
            image_np = Image.fromarray(image_np)
            image_np.save(os.path.join(DATA_PATH, f'renders/{ii+t}.png'))
            
            image_np_mujo = (mujoco_images_target1[ii] * 255).permute(1, 2, 0).to(torch.uint8)
            # blending two images together using a factor of 0.5
            image_np = image_np_render * 0.5 + image_np_mujo * 0.5
            image_np = image_np.to(torch.uint8)
            image_np = image_np.detach().cpu().numpy()
            # print(image_np.shape)
            image_np = Image.fromarray(image_np)
            image_np.save(os.path.join(DATA_PATH, f'blends/{ii+t}.png'))
        
        
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
        
        
        
        # threshold = 0.5
        # gaussian_masks_binary = (gaussian_masks > threshold).float()  # Convert to 0s and 1s
        # ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
        # gaussian_masks_scaled = (gaussian_masks_binary * 255).byte()  # uint8 format
        # ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()

        # # Step 3: Convert to NumPy
        # gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        # ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 4: Save masks as images
        # os.makedirs('masks_inspection', exist_ok=True)
        # for i in range(gaussian_masks_np.shape[0]):
        #     # Save Gaussian mask
        #     gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
        #     cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
        #     # print(f"Saved {gaussian_filename}")

        #     # Save Ground Truth mask
        #     gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
        #     cv2.imwrite(gt_filename, ground_truth_masks_np[i])
        #     # print(f"Saved {gt_filename}")
        
        # gaussian_depths_np = gaussian_depths.detach().cpu().numpy()
        # mujoco_depths_np = mujoco_depths_target.detach().cpu().numpy()
        # gaussian_depths_np *= 40
        # mujoco_depths_np *= 40
        # os.makedirs('depths_inspection', exist_ok=True)
        # for i in range(gaussian_masks_np.shape[0]):
        #     # Save Gaussian mask
        #     gaussian_filename = f"depths_inspection/gaussian_depth_{i}.png"
        #     cv2.imwrite(gaussian_filename, gaussian_depths_np[i])
        #     # print(f"Saved {gaussian_filename}")

        #     # Save Ground Truth mask
        #     gt_filename = f"depths_inspection/ground_truth_depth_{i}.png"
        #     cv2.imwrite(gt_filename, mujoco_depths_np[i])
        #     # print(f"Saved {gt_filename}")
        
        # print(gaussian_depths.max(), gaussian_depths.min(), mujoco_depths_target.max(), mujoco_depths_target.min())
        # print(gaussian_depths.shape, 'gaussian_depths')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(mujoco_images_target.shape, 'mujoco_images_target')
        # print(mujoco_mask_target.shape, 'mujoco_mask_target')
        
        # print(mujoco_mask_target.shape, 'mujoco mask shape')
        # print(mujoco_images_target1.shape, 'mujoco image shape')
        # mujoco_images_target1 = mujoco_images_target1 * mujoco_mask_target.unsqueeze(1)
        # mse_loss = F.mse_loss(mujoco_images_target1, gaussian_tensors, reduction='none')
        # l2_diff = mse_loss.mean(dim=(1, 2, 3))
        
        # IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
        # depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
        # depth_diff = depth_loss.mean()
        # if step % 10 == 0 or step == optimization_steps - 1:
        #     print(IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff')
        #     if dino_orig:
        #         print(loss_direct_feat.item(), 'loss_direct_feat')
        #     print(l2_diff.sum(), 'l2')
        
        
        # total_loss = l2_diff.sum() + 3 * IoU_loss + 1 * depth_diff + 1 * loss_feat
        # total_loss = l2_diff.sum() + 1 * IoU_loss
        # total_loss = IoU_loss
        # best: 10 depth_diff + 5 IoU_loss + 5 loss_direct_feat
        # total_loss = 5 * depth_diff + 10 * IoU_loss + 10 * loss_feat
        # total_loss.backward()

        # all_losses = l2_diff.detach().cpu().numpy().tolist()
        # all_images = [torch.clamp(tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() for tensor in gaussian_tensors]

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()    

        # if step % 10 == 0 or step == optimization_steps - 1:
        #     grid_images = np.zeros((360 * grid_size, 640 * grid_size, 3))
        #     for i in range(grid_size):
        #         for j in range(grid_size):
        #             idx = i * grid_size + j
        #             if idx < len(all_images):
        #                 grid_images[i*360:(i+1)*360, j*640:(j+1)*640] = all_images[idx]

            # best_idx = np.argmin(all_losses)
            # best_camera = all_cameras[best_idx]
            # best_joint_pose = all_joint_poses[best_idx]
            # angle_diff, trans_diff = all_cameras[0].transformation_distance()
            # tcp_distance = all_cameras[0].tcp_translation(tcp)
            # print('angle_diff', angle_diff, 'trans_diff', trans_diff, 'tcp: ',tcp_distance)
            
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

            

    # Final yield with only the best result
    # best_idx = np.argmin(all_losses)
    # best_image = all_images[best_idx]
    # best_camera = all_cameras[best_idx]
    # best_joint_pose = all_joint_poses[best_idx]

    # Extract final parameters for the best camera
    # final_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
    # final_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
    # final_camera_params = extract_camera_parameters(final_extrinsic.T)
    # print('Camera results robot to world transformation:')
    # # for i in range(len(all_cameras)):
    # print(all_cameras[0].robot_to_world())
    # # print(best_camera.robot_to_world())
    # params = all_cameras[0].get_params()
    # angle_diff, trans_diff = all_cameras[0].transformation_distance()
    # tcp_distance = all_cameras[0].tcp_translation(tcp)
    # print('angle_diff', angle_diff, 'trans_diff', trans_diff, 'tcp: ',tcp_distance.item())
    
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
        
    
    # final_params = final_joint_angles + [
    #     final_camera_params['azimuth'],
    #     final_camera_params['elevation'],
    #     final_camera_params['distance']
    # ]

    # rounded_final_params = [round(param, 2) for param in final_params]

    # yield (best_image, *rounded_final_params)
    # yield (all_images[Display], *rounded_final_params)
    return []


if __name__ == "__main__":
    Display = 0
    BATCH_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/rh20t_data/'
    FRANKA = True
    UR5 = False
    dino_vit = False
    dino_orig = False
    Optimal_Camera_Param = {'w': torch.tensor([ 0.0267, -0.028, -0.0195]), 'v': torch.tensor([-0.042,  0.0215, -0.006])}
    Optimal_Camera_Param = {'w': torch.tensor([-0.0049, -0.0286, -0.0441]), 'v': torch.tensor([ 0.0121,  0.0143, -0.0107])}
    scene_names = os.listdir(BATCH_PATH)
    scene_names = sorted(scene_names)
    print(scene_names)
    all_params = []

    # extractor = ViTExtractor('dino_vits8', 4, device='cuda')
    
    for scene_name in scene_names:
        print(scene_name)
        # if scene_name != 'f0461559_task_0123_user_0007_scene_0002_cfg_0005':
        #     continue
        DATA_PATH = os.path.join(BATCH_PATH, scene_name)
        
        image_list = []
        depth_list = []
        mask_list = []
        grippers = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
        joints = np.load(os.path.join(DATA_PATH, 'joints.npy'))
        tcp = np.load(os.path.join(DATA_PATH, 'tcp.npy')) # (3,)
        if UR5:
            fingers = np.stack([grippers, grippers, grippers, grippers, grippers, grippers, grippers, grippers], axis=1) # for ur5
        elif FRANKA:
            grippers = grippers * 0
            fingers = np.stack([grippers, grippers], axis=1) # for franka
        if UR5:
            offset_angle = 90 * np.pi / 180
            joints[:, 0] -= offset_angle
        joints = np.concatenate([joints, fingers], axis=1)
        print('joints: ', joints.shape)
        image_names = os.listdir(os.path.join(DATA_PATH, 'images'))
        image_names = sorted(image_names)
        for image_name in image_names:
            # print('image_name', image_name)
            if '.DS_Store' in image_name:
                continue
            image_path = os.path.join(DATA_PATH, 'images', image_name)
            img = Image.open(image_path)
            img = np.array(img)
            image_list.append(img)
        image_list = np.stack(image_list)
            
        depth_names = os.listdir(os.path.join(DATA_PATH, 'depths'))
        depth_names = sorted(depth_names)
        for depth_name in depth_names:
            # print('depth_name', depth_name)
            if '.DS_Store' in depth_name:
                continue
            depth_path = os.path.join(DATA_PATH, 'depths', depth_name)
            depth = np.load(depth_path)
            depth_list.append(depth)
        depth_list = np.stack(depth_list)
        
        # mask_names = os.listdir(os.path.join(DATA_PATH, 'masks'))
        # mask_names = sorted(mask_names)
        # for mask_name in mask_names:
        #     # print('mask_name', mask_name)
        #     if '.DS_Store' in mask_name:
        #         continue
        #     mask_path = os.path.join(DATA_PATH, 'masks', mask_name)
        #     mask = Image.open(mask_path)
        #     mask = np.array(mask)
        #     mask_list.append(mask)
        # mask_list = np.stack(mask_list)
        # print(mask_list.shape, 'masks ensure not 3 channels')
        # assert(depth_list.shape[0] == image_list.shape[0] == grippers.shape[0] == joints.shape[0] == mask_list.shape[0])
        # print('frames: ', image_list.shape[0])
        
        intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
        extrinsics = np.load(os.path.join(DATA_PATH, 'extrinsics.npy'))
        
        Optimize_list = []
        for timestep in range(5):
            Optimize_list.append(timestep)
        # Optimize_list = [0]
        
        optimize_camera = False
        optimize_joints = False
        camera_lr = 0.005
        joints_lr = 0.001
        optimization_steps = 1
        powerful_optimize_dropdown = "Disabled"
        noise_input = 0.01
        num_inits_input = 1
        print(len(image_list))
        for t in range(len(image_list)):
            image_list1 = image_list[t:t+1]
            params = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        powerful_optimize_dropdown, noise_input, num_inits_input)
        all_params.append(params)
    # last_elements = [param[-1] for param in all_params]
    # print(last_elements)
    # print(all_params[:, -1])