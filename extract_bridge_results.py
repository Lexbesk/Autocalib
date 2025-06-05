import pickle
import numpy as np
import os
import json

ROOT_PATH = '/data/group_data/katefgroup/datasets/bridge_chenyu/numpy/rss'
OUT_PATH = '/data/group_data/katefgroup/datasets/bridge_chenyu/numpy/rss'

all_data = {}

kitchen_names = os.listdir(ROOT_PATH)
kitchen_names.sort()
for kitchen_name in kitchen_names:
    kitchen_path = os.path.join(ROOT_PATH, kitchen_name)
    if not os.path.isdir(kitchen_path):
        continue
    task_names = os.listdir(kitchen_path)
    task_names.sort()
    
    for task_name in task_names:
        task_path = os.path.join(kitchen_path, task_name)
        episode_names = os.listdir(task_path) # 00, 01, ...
        episode_names.sort()
        for episode_name in episode_names:
            episode_path = os.path.join(task_path, episode_name)
            dates = os.listdir(episode_path)
            dates.sort()
            for date in dates:
                date_path = os.path.join(episode_path, date)
                print(date_path)
                
                # Check if the file exists
                if os.path.exists(os.path.join(date_path, 'results.pkl')):
                    with open(os.path.join(date_path, 'results.pkl'), 'rb') as f:
                        data = pickle.load(f)
                    
                    print(data.keys())
                    extrinsic_intrinsic = {}
                    for traj_id in data.keys():
                        traj_path = os.path.join(date_path, f'traj{traj_id}')
                        print(traj_path)
                        extrinsic_intrinsic['w2c'] = data[traj_id]['extrinsics']
                        extrinsic_intrinsic['intrinsic'] = data[traj_id]['intrinsics']
                        print(extrinsic_intrinsic)
                        all_data[traj_path] = extrinsic_intrinsic
print(all_data.keys())
with open(os.path.join(OUT_PATH, "w2c_intrinsic.json"), "w") as file:
    json.dump(all_data, file, indent=4)
                        
                        