import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
import pickle
import numpy as np
import pandas as pd
import glob
import os
import joblib


class EgobodyDataset(Dataset):
    def __init__(self, original_dataset_dir, pare_result_dir, preprocessed_dir, split, cfg=None, training=True,
                 seq_len=64, ntime_per_epoch=10000, first_n=0):
        self.cfg = cfg
        self.original_dataset_dir = original_dataset_dir
        self.pare_result_dir = pare_result_dir
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.training = training
        self.seq_len = seq_len
        self.ntime_per_epoch = ntime_per_epoch
        self.epoch_init_seed = None

        data_split_df = pd.read_csv(os.path.join(self.original_dataset_dir, 'data_splits.csv'))
        # TODO: hard coded for val on Mac
        if self.split == 'test':
            self.split = 'val'  # not really test during training
        self.split_list = list(data_split_df[self.split])
        while np.nan in self.split_list:
            self.split_list.remove(np.nan)
        if first_n != 0:
            self.split_list = self.split_list[:first_n]

        self.data_pare_features, self.data_frame_visibility, self.data_joint_visibility = {}, {}, {}
        self.data_shape, self.data_pose = {}, {}
        self.load_data()

        self.sequences = list(self.data_frame_visibility.keys())
        # compute sampling probablity
        self.seq_lengths = np.array([x.shape[0] for x in self.data_frame_visibility.values()])
        if cfg is not None and cfg.seq_sampling_method == 'length':
            self.seq_prob = self.seq_lengths / self.seq_lengths.sum()
        else:
            self.seq_prob = None

    def __len__(self):
        return self.ntime_per_epoch // self.seq_len

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len

    def load_data(self):
        for recording in self.split_list:
            data = joblib.load(os.path.join(self.pare_result_dir, self.split, recording, "pare_results.pkl"))
            self.data_pare_features.update({recording: data['point_local_feat']})
            data = joblib.load(os.path.join(self.preprocessed_dir, self.split, f'{recording}.pkl'))
            self.data_joint_visibility.update({recording: data['joints']['j2d_visible_smpl_order']})
            self.data_frame_visibility.update({recording: data['frames']['frame_visible']})
            self.data_pose.update({recording: data['smpl_parameters']['pose_cam']})
            self.data_shape.update({recording: data['smpl_parameters']['shape']})

    def __getitem__(self, idx):
        if self.epoch_init_seed is None:
            # the above step is necessary for lightning's ddp parallel computing because each node gets a subset (idx) of the dataset
            self.epoch_init_seed = (np.random.get_state()[1][0] * len(self) + idx) % int(1e8)
            np.random.seed(self.epoch_init_seed)
            # print('epoch_init_seed', self.epoch_init_seed)

        success = False
        while not success:
            sind = np.random.choice(len(self.sequences), p=self.seq_prob)
            seq = self.sequences[sind]
            pare_feature, joint_visibility, frame_visibility = self.data_pare_features[seq], \
                self.data_joint_visibility[seq], self.data_frame_visibility[seq]
            shape, pose = self.data_shape[seq], self.data_pose[seq]

            if self.seq_len <= self.data_frame_visibility[seq].shape[0]:
                possible_start_frame = self.find_start_frame(frame_visibility, preserve_first_n=10)
                if sum(possible_start_frame) != 0:
                    possible_start_idx = np.where(possible_start_frame)[0]
                    fr_start = np.random.choice(possible_start_idx)
                    seq_pare_feature = pare_feature[fr_start: fr_start + self.seq_len].astype(np.float32)
                    frame_loss_mask = np.ones((self.seq_len, 1)).astype(np.float32)  # TODO: check this!
                    eff_seq_len = self.seq_len  # effective seq
                    seq_joint_visibility = joint_visibility[fr_start: fr_start + self.seq_len].astype(np.float32)
                    seq_frame_visibility = frame_visibility[fr_start: fr_start + self.seq_len].astype(np.float32)
                    seq_shape = shape[fr_start: fr_start + self.seq_len].astype(np.float32)
                    seq_pose = pose[fr_start: fr_start + self.seq_len].astype(np.float32)
                    success = True
                else:
                    print(f'{seq} do not have a possible start frame!')
                    # raise NotImplementedError
            else:
                print(f'{seq} too short!')
                # raise NotImplementedError

        data = {
            'point_local_feat': seq_pare_feature,
            'seq_name': seq,
            'frame_loss_mask': frame_loss_mask,
            'fr_start': fr_start,
            'eff_seq_len': eff_seq_len,
            'joint_mask': seq_joint_visibility,
            'frame_mask': seq_frame_visibility,
            'shape_gt': seq_shape,
            'pose_gt': seq_pose,
            # 'seq_ind': sind,
            # 'idx': idx,
        }
        # self.generate_mask(data)  # seq_frame_visibility already serves as mask

        # TODO: check if need to smooth the PARE features?

        return data

    def find_start_frame(self, visibility, preserve_first_n=10):
        result = np.zeros_like(visibility)
        for i in range(visibility.shape[0] - self.seq_len + 1):
            if sum(visibility[i:i + preserve_first_n]) == preserve_first_n:
                result[i] = 1
        return result


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    egobody_dataset_dir = 'dataset/egobody_dataset'
    egobody_pare_predict_dir = 'dataset/egobody_pare_predicts'
    egobody_preprocessed_dir = 'dataset/egobody_preprocessed'

    dataset = EgobodyDataset(egobody_dataset_dir, egobody_pare_predict_dir, egobody_preprocessed_dir,
                             'train', seq_len=200, first_n=4)
    print(f'dataset has {len(dataset)} data')

    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(batch['pose'].shape)
