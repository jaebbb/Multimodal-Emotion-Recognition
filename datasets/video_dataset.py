import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_video
import os
import glob
import yaml
import re

from utils.video_read import read_video_frames_from_dir


class VideoDataset(Dataset):
    def __init__(self, video_list_file, video_transform=None, face_video_transform=None, audio_transform=None):
        """
        Args:
            video_list_file (str): Path to yaml or json file that contains "video: target"
            video_transform
            face_video_transform
            audio_transform

        __getitem__:
            Args:
                index (int)
            Returns:
                sample (Dict): Python dictionary contains vframes, face_vframes, aframes, info, label
                    vframes (C, T, H, W)
                    face_vframes (C, T, H, W)
        """
        self.video_list_file = video_list_file
        with open(video_list_file, 'r') as f:
            self.video_list = yaml.load(f, yaml.Loader)

        self.video_transform = video_transform
        self.face_video_transform = face_video_transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_file_and_target = list(self.video_list.items())[index]

        # Read Video and Audio
        vframes, aframes, info = read_video(video_file_and_target[0])
        vframes = vframes.permute((3, 0, 1, 2)).numpy()

        aframes = aframes.numpy()

        # Read Face Frames
        face_frame_dir = self.get_face_dir(video_file_and_target[0])
        face_vframes = read_video_frames_from_dir(face_frame_dir, (224, 224))

        # Label
        label = self.emotion_to_number(video_file_and_target[1])

        if self.video_transform is not None:
            vframes = self.video_transform(vframes)

        if self.face_video_transform is not None:
            face_vframes = self.face_video_transform(face_vframes)

        if self.audio_transform is not None:
            aframes = self.audio_tranfrom(aframes)

        sample = {'vframes': vframes, 'face_vframes': face_vframes, 'aframes': aframes,
                  'info': info, 'label': label}

        return sample

    def get_face_dir(self, video_file):
        face_dir = re.sub('\.mp4', '', video_file)
        face_dir = re.sub('train', 'train_face', face_dir)
        face_dir = re.sub('val', 'val_face', face_dir)
        face_dir = re.sub('test', 'test_face', face_dir)
        return face_dir

    def emotion_to_number(self, label):
        return {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6, None: -1}.get(label)
