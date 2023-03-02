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
from scipy.interpolate import interp1d


class EgobodyDataset(Dataset):
    def __init__(self, original_dataset_dir, pare_result_dir, preprocessed_dir, split, cfg=None, training=True,
                 seq_len=64, ntime_per_epoch=10000, first_n=0, random=False, use_mask=True):
        self.cfg = cfg
        self.original_dataset_dir = original_dataset_dir
        self.pare_result_dir = pare_result_dir
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.training = training
        self.seq_len = seq_len
        self.ntime_per_epoch = ntime_per_epoch
        self.epoch_init_seed = None
        self.preserve_first_n = 10
        self.preserve_last_n = 10
        self.random = random
        self.use_mask = use_mask
        self.min_mask, self.max_mask = 10, 30

        data_split_df = pd.read_csv(os.path.join(self.original_dataset_dir, 'data_splits.csv'))
        # TODO: hard coded for val on Mac
        if self.split == 'test':
            self.split = 'val'  # not really test during training
        self.split_list = list(data_split_df[self.split])
        while np.nan in self.split_list:
            self.split_list.remove(np.nan)
        if first_n != 0:
            self.split_list = self.split_list[:first_n]
        excluded_list = ['recording_20210911_S07_S06_03', 'recording_20210911_S07_S06_02']
        for x in excluded_list:
            if x in self.split_list:  # no visible frames
                self.split_list.remove(x)
        self.data_pare_features, self.data_frame_visibility, self.data_joint_visibility = {}, {}, {}
        self.data_shape, self.data_pose = {}, {}
        self.load_data()

        self.sequences = list(self.data_frame_visibility.keys())
        self.all_possible_frames = {}
        self.calculate_all_possible_frames()
        # compute sampling probablity
        self.seq_lengths = np.array([x.shape[0] for x in self.all_possible_frames.values()])
        if cfg is not None and cfg.seq_sampling_method == 'length':
            self.seq_prob = self.seq_lengths / self.seq_lengths.sum()
        else:
            self.seq_prob = None

    def __len__(self):
        if self.random:
            return self.ntime_per_epoch // self.seq_len
        else:
            return self.seq_lengths.sum()

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
        if self.random:
            if self.epoch_init_seed is None:
                # the above step is necessary for lightning's ddp parallel computing because each node gets a subset (idx) of the dataset
                self.epoch_init_seed = (np.random.get_state()[1][0] * len(self) + idx) % int(1e8)
                np.random.seed(self.epoch_init_seed)
                # print('epoch_init_seed', self.epoch_init_seed)

            sind = np.random.choice(len(self.sequences), p=self.seq_prob)
            seq = self.sequences[sind]

            possible_start_idx = self.all_possible_frames[seq]
            fr_start = np.random.choice(possible_start_idx)
        else:
            sind = np.argmin(self.seq_lengths.cumsum() <= idx)
            seq = self.sequences[sind]
            fr_start = self.all_possible_frames[seq][idx] if sind == 0 \
                else self.all_possible_frames[seq][idx - self.seq_lengths.cumsum()[sind - 1]]

        pare_feature, joint_visibility, frame_visibility = self.data_pare_features[seq], \
            self.data_joint_visibility[seq], self.data_frame_visibility[seq]
        shape, pose = self.data_shape[seq], self.data_pose[seq]

        seq_pare_feature = pare_feature[fr_start: fr_start + self.seq_len].copy().astype(np.float32)
        frame_loss_mask = np.ones((self.seq_len, 1)).astype(np.float32)  # TODO: check this!
        eff_seq_len = self.seq_len  # effective seq
        seq_joint_visibility = joint_visibility[fr_start: fr_start + self.seq_len].copy().astype(np.float32)
        seq_frame_visibility = frame_visibility[fr_start: fr_start + self.seq_len].astype(np.float32)
        seq_shape = shape[fr_start: fr_start + self.seq_len].astype(np.float32)
        seq_pose = pose[fr_start: fr_start + self.seq_len].astype(np.float32)

        if self.use_mask:
            for joint in range(seq_joint_visibility.shape[1]):
                if sum(seq_joint_visibility[:, joint]) == seq_joint_visibility[:, joint].size:  # all joint visible
                    drop_len = np.random.randint(self.min_mask, self.max_mask + 1)
                    start_fr_min = self.preserve_first_n
                    start_fr_max = min(self.seq_len - drop_len + 1 - self.preserve_last_n, self.seq_len)
                    start_fr = np.random.randint(start_fr_min, start_fr_max)
                    end_fr = min(start_fr + drop_len, self.seq_len)

                    seq_joint_visibility[start_fr: end_fr, joint] = 0.

                    vis_ind = np.where(seq_joint_visibility[:, joint])[0]
                    f = interp1d(vis_ind.astype(np.float32), seq_pare_feature[seq_joint_visibility[:, joint]==1, :, joint],
                                 axis=0, assume_sorted=True)
                    seq_pare_feature[:, :, joint] = f(np.arange(self.seq_len, dtype=np.float32))

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

    def find_start_frame(self, visibility, preserve_first_n=10, preserve_last_n=10):
        result = np.zeros_like(visibility)
        for i in range(visibility.shape[0] - self.seq_len + 1):
            if (sum(visibility[i:i + preserve_first_n]) == preserve_first_n) and \
                    (sum(visibility[i + self.seq_len - preserve_last_n:i + self.seq_len]) == preserve_last_n):
                result[i] = 1
        return result

    def calculate_all_possible_frames(self):
        for seq in self.sequences:
            frame_visibility = self.data_frame_visibility[seq]
            possible_start_frame = self.find_start_frame(frame_visibility,
                                                         preserve_first_n=self.preserve_first_n,
                                                         preserve_last_n=self.preserve_last_n)
            possible_start_idx = np.where(possible_start_frame)[0]
            self.all_possible_frames.update({seq: possible_start_idx})


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    egobody_dataset_dir = 'dataset/egobody_dataset'
    egobody_pare_predict_dir = 'dataset/egobody_pare_predicts'
    egobody_preprocessed_dir = 'dataset/egobody_preprocessed'

    dataset = EgobodyDataset(egobody_dataset_dir, egobody_pare_predict_dir, egobody_preprocessed_dir,
                             'train', seq_len=50, first_n=4)
    print(f'dataset has {len(dataset)} data')

    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    dataset.calculate_all_possible_frames()
    print(dataset.all_possible_frames)
    print('ok')
