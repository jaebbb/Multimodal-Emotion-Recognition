import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_video
import os
import glob
import yaml


class VideoDataset(Dataset):
    def __init__(self, video_list_file, video_transform=None, audio_transform=None):
        """
        Args:
            video_list_file (str): Path to yaml or json file that contains "video: target"
            video_transform
            audio_transform

        __getitem__:
            Args:
                index (int)
            Returns:
                sample (Dict): Python dictionary contains vframes, aframes, info, label
        """
        self.video_list_file = video_list_file
        with open(video_list_file, 'r') as f:
            self.video_list = yaml.load(f, yaml.Loader)

        self.video_transform = video_transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_file_and_target = list(self.video_list.items())[index]
        vframes, aframes, info = read_video(video_file_and_target[0])
        label = video_file_and_target[1]

        if self.video_transform is not None:
            vframes = self.video_transform(vframes)

        if self.audio_transform is not None:
            aframes = self.audio_tranfrom(aframes)

        sample = {'vframes': vframes, 'aframes': aframes,
                  'info': info, 'label': label}

        return sample
