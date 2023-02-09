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

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='glamr_dynamic')
parser.add_argument('--img_folder',
                    default='dataset/egobody_dataset/egocentric_color/recording_20210911_S03_S08_02/2021-0911-172407/PV')
parser.add_argument('--out_dir', default='out/glamr_dynamic/egobody/recording_20210911_S03_S08_02')
parser.add_argument('--pose_est_dir', default='/Users/chuanlongzang/Projects/PARE/logs/'
                                              'recording_20210911_S03_S08_02_imgs/PV_/pare_results/*')
parser.add_argument('--gt_dir', default='/Users/chuanlongzang/Projects/Thesis/datasets/egobody_preprocessed/'
                                        'test/recording_20210911_S03_S08_02.pkl')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=1)
parser.add_argument('--save_video', action='store_true', default=False)
args = parser.parse_args()

cfg = Config(args.cfg, out_dir=args.out_dir)
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

cfg.save_yml_file(f'{args.out_dir}/config.yml')
log = create_logger(f'{cfg.log_dir}/log.txt')
grecon_path = f'{args.out_dir}/grecon'
render_path = f'{args.out_dir}/grecon_videos'
seq_name = args.out_dir.split('/')[-1]
os.makedirs(grecon_path, exist_ok=True)
os.makedirs(render_path, exist_ok=True)

pare_results = glob.glob(args.pose_est_dir)
pare_results = sorted(pare_results)


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


pare_results_path = f'{args.out_dir}/pare_results.pkl'
if osp.exists(pare_results_path):
    pare_results = pickle.load(open(pare_results_path, 'rb'))
else:
    pare_results = merge_results(pare_results)
    pickle.dump(pare_results, open(pare_results_path, 'wb'))

gt_result = joblib.load(args.gt_dir)
in_dict = {'est': pare_results, 'gt': gt_result, 'gt_meta': dict(), 'seq_name': seq_name}
grecon_model = model_dict[cfg.grecon_model_name](cfg, device, log)  # global recon model
out_dict = grecon_model.optimize(in_dict)

out_file = f'{grecon_path}/{seq_name}_result.pkl'
pickle.dump(out_dict, open(out_file, 'wb'))


print('ok')
