import torch
from torch import nn
from models.s3d import S3D


class NaburangiV1(nn.Module):
    def __init__(self):
        self.face_tiny_detection = None
        self.face_s3d = S3D(end_point='avg_pool3d')
        self.s3d = S3D(end_point='avg_pool3d')
        # self.audio_model = None
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 100)
        self.linear3 = nn.Linear(7, 512)

    def forward(self, v_x, a_x):
        face_x = self.face_tiny_detection(v_x)
        face_feature = self.face_s3d(face_x)  # (batch, 1024)
        v_feature = self.s3d(v_x)  # (batch, 1024)
        # a_feature = self.audio_model(a_x)
        feature = torch.cat((face_feature, v_feature), dim=1)
        x = self.linear1(feature)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
