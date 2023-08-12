import os, sys

sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import glob
import torch
import numpy as np
import pickle
import cv2 as cv
import shutil
import argparse
from lib.utils.log_utils import create_logger
from lib.utils.vis import get_video_num_fr, get_video_fps, hstack_video_arr, get_video_width_height, video_to_images
from global_recon.utils.config import Config
import joblib
import pandas as pd
import subprocess
import time
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='dataset/egobody_dataset')
parser.add_argument('--preprocess_out_dir', default='dataset/egobody_pare_predicts')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', default='all')  # all means train + val
parser.add_argument('--from_to_index', default='0-0')  # for parallel jobs in slurm
args = parser.parse_args()

if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

data_split_df = pd.read_csv(os.path.join(args.dataset_root, 'data_splits.csv'))
train_split_list = list(data_split_df['train'])
val_split_list = list(data_split_df['val'])
excluded_list = ['recording_20210911_S07_S06_03']
conda_path = os.environ["CONDA_PREFIX"].split('/envs')[0]
while np.nan in train_split_list:
    train_split_list.remove(np.nan)
while np.nan in val_split_list:
    val_split_list.remove(np.nan)

def merge_results(file_list):
    results = dict()
    for i in range(len(file_list)):
        pare_result = joblib.load(file_list[i])
        if results:
            for key, value in pare_result.items():
                results[key] = np.concatenate((results[key], value), axis=0)
        else:
            results = pare_result.copy()
    return results


def get_joint_visibility_smpl_order(results):
    joint_visibility = np.ones(results['smpl_joints2d_smpl_order'].shape[:-1])
    for i in range(results['smpl_joints2d_smpl_order'].shape[0]):
        for j in range(results['smpl_joints2d_smpl_order'].shape[1]):
            u, v = results['smpl_joints2d_smpl_order'][i, j]
            if u <= 0 or u >= 1920 or v <= 0 or v >= 1080:
                joint_visibility[i][j] = 0.0
    results['vis_joints'] = joint_visibility
    return results

period_dict = {}
if args.split == 'train':
    period_dict = {'train': train_split_list}
elif args.split == 'val':
    period_dict = {'val': val_split_list}
elif args.split == 'all':
    period_dict = {'train': train_split_list,
                   'val': val_split_list}
else:
    print('Not recognized split!')

from_index, to_index = map(int, args.from_to_index.split('-'))
if from_index == 0 and to_index == 0:
    period_dict = period_dict
else:
    period_dict = {k: v[from_index: to_index] for k, v in period_dict.items()}

for period, recording_list in period_dict.items():
    recording_num = len(recording_list)

    for i in tqdm.tqdm(range(recording_num)):
        recording_name = recording_list[i]
        if recording_name in excluded_list:
            print(f'Ignoring {recording_name}')
            continue
        else:
            print(f'Preprocessing {recording_name}.')

        if not osp.exists(f'./PARE/logs/{period}/{recording_name}/PV_/pare_results'):
            image_folder = os.path.join(
                glob.glob(os.path.join(args.dataset_root, 'egocentric_color', recording_name, '202*'))[0], 'PV')
            bbox_file = f'../dataset/egobody_preprocessed/{period}/{recording_name}.pkl'
            cmd = f'{conda_path}/envs/pare-env/bin/python scripts/detection_egobody.py --image_folder ../{image_folder} ' \
                  f'--output_folder logs/{period}/{recording_name} --no_render --bbox_file {bbox_file}'
            subprocess.run(cmd.split(' '), cwd='./PARE')

        # pare returns per frame result, need to merge them to a video
        pose_est_dir = f'PARE/logs/{period}/{recording_name}/PV_/pare_results/*'
        pare_results = glob.glob(pose_est_dir)
        pare_results = sorted(pare_results)

        out_dir = os.path.join(args.preprocess_out_dir, period, recording_name)
        os.makedirs(out_dir, exist_ok=True)
        pare_results_path = f'{out_dir}/pare_results.pkl'
        if osp.exists(pare_results_path):
            pare_results = pickle.load(open(pare_results_path, 'rb'))
        else:
            pare_results = merge_results(pare_results)
            if 'vis_joints' not in pare_results.keys():
                pare_results = get_joint_visibility_smpl_order(pare_results)
                # pare_results.pop('smpl_joints2d_smpl_order', None)
            try:
                with open(pare_results_path, 'wb') as f:
                    pickle.dump(pare_results, f)
            except OverflowError:
                print(f'Recording {recording_name} occurs an overflow error, please consider reducing file size!')
                with open(pare_results_path, 'wb') as f:
                    pickle.dump(pare_results, f, protocol=4)
            time.sleep(1)  # make sure pickle is saving properly (for large files)

print('ok')
