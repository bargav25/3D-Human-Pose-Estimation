import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
import logging

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.model import * 

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import *
from tqdm import tqdm


def fetch(subjects, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():

            poses_2d = keypoints[subject][action]

            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None


    return out_camera_params, out_poses_3d, out_poses_2d

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = 'checkpoint'
    
    # Load dataset
    dataset_path = 'data/data_3d_h36m.npz'
    dataset = Human36mDataset(dataset_path)

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    keypoints = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)

    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = 'S1,S5,S6,S7,S8'.split(',')
    subjects_test = 'S9,S11'.split(',')

    cameras_train, poses_train, poses_train_2d = fetch(subjects_train)

    batch_size = 512
    num_input_frames = 81

    receptive_field = num_input_frames 

    pad = (receptive_field -1) // 2 # Padding on each side of the input vid.

    train_generator = ChunkedGenerator(batch_size, None, poses_train, poses_train_2d, 1,
                                        pad=pad, causal_shift=0, shuffle=True, augment=True,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    # for _, batch_3d, batch_2d in train_generator.next_epoch():
    #     print(batch_3d.shape)
    #     print(batch_2d.shape)
    #     break

    
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test)

    test_generator = UnchunkedGenerator(None, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=0, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    torch.cuda.empty_cache()

    min_loss = 100000
    width = cam['res_w']
    height = cam['res_h']
    num_joints = keypoints_metadata['num_joints']


    input_dim = 2
    d_model = 48
    num_heads = 8
    num_layers = 6
    dropout = 0.1
    seq_length = 81

    model_pos = LSTM_PoseNet(num_joints, num_frames=receptive_field, input_dim=2, output_dim=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pos = model_pos.to(device)

    optimizer = optim.Adam(model_pos.parameters(), lr=0.001)
    criterion = mpjpe

    num_epoch = 15
    no_eval = False

    lr = 0.0001
    lr_decay = 0.99
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    initial_momentum = 0.1
    final_momentum = 0.001

    for epoch in range(num_epoch):
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos.train()

        for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')) # [512, 1, 17, 3]
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [512, 3, 17, 2]

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()
            
            predicted_3d_pos = model_pos(inputs_2d)

            # print(predicted_3d_pos.size())
            # print(inputs_3d.size())

            loss_3d_pos = criterion(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()

            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos

            loss_total.backward()

            optimizer.step()
            torch.cuda.empty_cache()
            del inputs_2d, inputs_3d, loss_3d_pos, predicted_3d_pos

        losses_3d_train.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()

        with torch.no_grad():
            model_pos.load_state_dict(model_pos.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            N = 0
            if not no_eval:
                # Evaluate on test set
                for _, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32')) # [1, 2356, 17, 3]
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [1, 2358, 17, 2]

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d) # [2356, 3, 17, 2] 
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
            
                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                        inputs_3d = inputs_3d.cuda()

                    inputs_3d[:, :, 0] = 0
                    

                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                                joints_right + joints_left]

                    predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                    keepdim=True)

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    torch.cuda.empty_cache()

                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    del inputs_2d, inputs_2d_flip, inputs_3d, loss_3d_pos, predicted_3d_pos, predicted_3d_pos_flip
                    torch.cuda.empty_cache()

                losses_3d_valid.append(epoch_loss_3d_valid / N)

        elapsed = (time() - start_time) / 60

        if no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1] * 1000))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        #### save best checkpoint
        best_chk_path = os.path.join(checkpoint, 'best_epoch.bin'.format(epoch))
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, best_chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if epoch > 3:

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
    #         plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join('loss_curves', 'loss_3d.png'))
            plt.close('all')
