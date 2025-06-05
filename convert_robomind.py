
import os

# tmp_dir = os.path.join(os.getcwd(), 'tmp')
# os.makedirs(tmp_dir, exist_ok=True)
# tempfile.tempdir = tmp_dir
# print(f"Created temporary directory: {tmp_dir}")
# os.environ['TMPDIR'] = tmp_dir
# if 'notebooks' not in os.listdir(os.getcwd()):
#     os.chdir('../')
    

import numpy as np
from PIL import Image
from tqdm import tqdm


"""
python convert_robomind.py
"""

if __name__ == "__main__":
    BATCH_PATH = '/data/group_data/katefgroup/datasets/robomind/robomind_chenyu/auto_calibration/both_pour_water'
    EXT_PATH = '/data/group_data/katefgroup/datasets/robomind/robomind_chenyu/robomind_extract_2'
    # BATCH_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/robomind
    
    TRAJ_PATHs = [d for d in os.listdir(BATCH_PATH) if os.path.isdir(os.path.join(BATCH_PATH, d))]
    TRAJ_PATHs = sorted(TRAJ_PATHs)
    scene_count = -1
    for i, traj_name in enumerate(TRAJ_PATHs):
        print(i, traj_name)
        TRAJ_PATH = os.path.join(BATCH_PATH, traj_name)
        camera_names = [d for d in os.listdir(TRAJ_PATH) if os.path.isdir(os.path.join(TRAJ_PATH, d))]
        camera_names = sorted(camera_names)
        for j, cam_name in enumerate(camera_names):
            print(j, cam_name)
            scene_count += 1
            scene_path = os.path.join(EXT_PATH, f'scene_{scene_count}')
            os.makedirs(scene_path, exist_ok=True)
            # if cam_name.split('_')[1] != 'right': # left right top 
            #     continue
            # if cam_name != 'camera_left':
            #     continue
            DATA_PATH = os.path.join(TRAJ_PATH, cam_name)
            
            image_list_whole = []
            depth_list_whole = []
            mask_list_whole = []
            grippers_whole = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
            joints_whole = np.load(os.path.join(DATA_PATH, 'joints.npy'))
            np.save(os.path.join(scene_path, 'grippers.npy'), grippers_whole)
            np.save(os.path.join(scene_path, 'joints.npy'), joints_whole)
            print(grippers_whole.shape, 'grippers', grippers_whole.max(), grippers_whole.min())
            print('joints_whole: ', joints_whole.shape)
            # print(joints)
            image_names = os.listdir(os.path.join(DATA_PATH, 'images'))
            image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
            image_folder = os.path.join(scene_path, 'images0')
            os.makedirs(image_folder, exist_ok=True)
            for image_name in tqdm(image_names):
            # for image_name in image_names:
                # print('image_name', image_name)
                image_path = os.path.join(DATA_PATH, 'images', image_name)
                img = Image.open(image_path)
                img.save(os.path.join(image_folder, str(int(image_name.split('.')[0])) + '.jpg'))
            intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
            np.save(os.path.join(scene_path, 'intrinsics.npy'), intrinsics)
            if cam_name.split('_')[1] == 'left':
                w2c = np.array([[
                    -0.81138510327202173,
                    -0.18410696539124252,
                    0.55476016392913308,
                    0.0],
                    [-0.57686105632703244,
                    0.40528076144758174,
                    -0.70921000140559831,
                    0.0],
                    [-0.094263120474080767,
                    -0.89546196440434933,
                    -0.43504291101406634,
                    0.0],
                    [0.59219642021433283,
                    0.52541880645681949,
                    0.74251763181186836,
                    1.0]]).T
            if cam_name.split('_')[1] == 'right':
                w2c = np.array([[0.83734427822375257, # for right view
                    -0.27765933958399935,
                    0.47091384654490714,
                    0.0],
                    [-0.54666720000213354,
                    -0.42041948190580292,
                    0.72415635858278826,
                    0.0],
                    [-0.003087420892972168,
                    -0.86380133733158437,
                    -0.50382310135244501,
                    0.0],
                    [-0.54788088926660228,
                    0.67045439724472256,
                    0.68404495102964147,
                    1.0]]).T
            if cam_name.split('_')[1] == 'top':
                w2c = np.array([[-0.25389173788236485, # for top view
                    0.86150880238723326,
                    -0.43970623016326232,
                    0.0],
                    [0.96252137951360806,
                    0.18022378814692969,
                    -0.2026622317186689,
                    0.0],
                    [-0.095349774065245474,
                    -0.47468091345164271,
                    -0.87497797171724323,
                    0.0],
                    [0.2258082244284767,
                    -0.54554129322083722,
                    1.5017372929195465,
                    1.0]]).T
            c2w = np.linalg.inv(w2c)
            gt_extrinsics = w2c
            extrinsics = gt_extrinsics
            np.save(os.path.join(scene_path, 'extrinsics0.npy'), extrinsics)