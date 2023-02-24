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
from global_recon.models import model_dict
from global_recon.vis.vis_grecon import GReconVisualizer
from global_recon.vis.vis_cfg import demo_seq_render_specs as seq_render_specs
from pose_est.run_pose_est_demo import run_pose_est_on_video
import joblib
import pandas as pd
import subprocess
import time
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='glamr_dynamic_kp2d_tts+')
parser.add_argument('--dataset_root', default='dataset/egobody_dataset')
parser.add_argument('--out_dir', default='out/glamr_dynamic/egobody_joint_motion_infiller')
parser.add_argument('--gt_dir', default='dataset/egobody_preprocessed/test')
parser.add_argument('--evl_num', default=5)  # 0 means all
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=1)
parser.add_argument('--save_video', action='store_true', default=False)
args = parser.parse_args()

cfg = Config(args.cfg, out_dir=args.out_dir)
cfg_name = args.cfg.split("_", 2)[-1]
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

data_split_df = pd.read_csv(os.path.join(args.dataset_root, 'data_splits.csv'))
test_split_list = list(data_split_df['test'])
conda_path = os.environ["CONDA_PREFIX"].split('/envs')[0]
while np.nan in test_split_list:
    test_split_list.remove(np.nan)


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


if args.evl_num == '0':
    args.evl_num = len(test_split_list)
for i in tqdm.tqdm(range(args.evl_num)):
    recording_name = test_split_list[i]
    print(f'Optimizing {recording_name}.')

    if not osp.exists(f'./PARE/logs/{recording_name}/PV_/pare_results'):
        image_folder = os.path.join(
            glob.glob(os.path.join(args.dataset_root, 'egocentric_color', recording_name, '202*'))[0], 'PV')
        bbox_file = f'../dataset/egobody_preprocessed/test/{recording_name}.pkl'
        cmd = f'{conda_path}/envs/pare-env/bin/python scripts/detection_egobody.py --image_folder ../{image_folder} ' \
              f'--output_folder logs/{recording_name} --no_render --bbox_file {bbox_file}'
        subprocess.run(cmd.split(' '), cwd='./PARE')

    # pare returns per frame result, need to merge them to a video
    pose_est_dir = f'PARE/logs/{recording_name}/PV_/pare_results/*'
    pare_results = glob.glob(pose_est_dir)
    pare_results = sorted(pare_results)

    out_dir = os.path.join(args.out_dir, recording_name)
    os.makedirs(out_dir, exist_ok=True)
    pare_results_path = f'{out_dir}/pare_results.pkl'
    if osp.exists(pare_results_path):
        pare_results = pickle.load(open(pare_results_path, 'rb'))
    else:
        pare_results = merge_results(pare_results)
        if 'vis_joints' not in pare_results.keys():
            pare_results = get_joint_visibility_smpl_order(pare_results)
        with open(pare_results_path, 'wb') as f:
            pickle.dump(pare_results, f)
        time.sleep(5)  # make sure pickle is saving properly (for large files)

    cfg.save_yml_file(f'{out_dir}/config_{cfg_name}.yml')
    log = create_logger(f'{out_dir}/log_{cfg_name}.txt')
    grecon_path = f'{out_dir}/grecon'
    render_path = f'{out_dir}/grecon_videos'
    seq_name = 'egobody'  # args.out_dir.split('/')[-1]

    out_file = f'{grecon_path}/{seq_name}_result_{cfg_name}.pkl'
    out_file_bo = f'{grecon_path}/{seq_name}_result_bo.pkl'  # before_optimization
    if osp.exists(out_file) and osp.exists(out_file_bo):
        print(f'{recording_name} already has been analysed!')
    else:
        os.makedirs(grecon_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)

        gt_dir = os.path.join(args.gt_dir, f'{recording_name}.pkl')
        gt_result = joblib.load(gt_dir)
        in_dict = {'est': pare_results, 'gt': gt_result, 'gt_meta': dict(), 'seq_name': seq_name}
        grecon_model = model_dict[cfg.grecon_model_name](cfg, device, log)  # global recon model
        if not osp.exists(out_file_bo):
            out_dict = grecon_model.optimize(in_dict, bo=True)
            pickle.dump(out_dict, open(out_file_bo, 'wb'))
        if not osp.exists(out_file):
            out_dict = grecon_model.optimize(in_dict, bo=False)
            pickle.dump(out_dict, open(out_file, 'wb'))

print('ok')
