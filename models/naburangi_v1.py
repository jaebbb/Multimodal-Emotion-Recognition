import torch
from torch import nn
from models.s3d import S3D


class NaburangiV1(nn.Module):
    def __init__(self):
        """
        forward:
            Args:
                v_x (Tensor(B, C, T, H, W)): Video frames
                face_v_x (Tensor(B, C, T, H, W)): Face video frames
                a_x (Tensor): Audio
            Returns:
                x (Tensor (B, 7))
        """
        super(NaburangiV1, self).__init__()
        self.face_s3d = S3D(end_point='avg_pool3d')
        self.s3d = S3D(end_point='avg_pool3d')
        # self.audio_model = None
        # self.linear1 = nn.Linear(2048, 512)
        # self.linear2 = nn.Linear(512, 100)
        # self.linear3 = nn.Linear(7, 512)
        self.fc = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=1, stride=1, bias=True),
            nn.Conv1d(512, 100, kernel_size=1, stride=1, bias=True),
            nn.Conv1d(100, 7, kernel_size=1, stride=1, bias=True))

    def forward(self, v_x, face_v_x, a_x):
        # Video
        v_feature = self.s3d(v_x)
        v_feature = v_feature.squeeze(4).squeeze(3)  # (batch, 1024, t)

        # Face
        face_feature = self.face_s3d(face_v_x)
        face_feature = face_feature.squeeze(4).squeeze(3)  # (batch, 1024, t)

        # a_feature = self.audio_model(a_x)
        feature = torch.cat((face_feature, v_feature), dim=1)

        # Classification
        x = self.fc(feature)
        x = torch.mean(x, 2)
        return x
