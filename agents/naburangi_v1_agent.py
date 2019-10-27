import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import datetime

from agents.base import BaseAgent
from models.naburangi_v1 import NaburangiV1
from datasets.video_dataset import VideoDataset
from utils.video_transforms import train_compose


class NaburangiV1Agent(BaseAgent):
    def __init__(self, config):
        super(NaburangiV1Agent, self).__init__(config)
        self.start_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.device = torch.device(self.config.device)
        self.current_epoch = 1
        # Data Loaders
        # Train Data Loaders
        if self.config.train.train_run is True:
            train_video_transform = train_compose(
                self.config.graph.model.max_seq_length, (224, 384))
            train_face_video_transform = train_compose(
                self.config.graph.model.max_seq_length, (224, 224))
            self.train_dataloader = DataLoader(VideoDataset(
                self.config.train.train_list_file, video_transform=train_video_transform, face_video_transform=train_face_video_transform), batch_size=self.config.train.train_batch_size, shuffle=True, num_workers=self.config.train.train_num_workers)

        # Val Data Loaders
        if self.config.val.val_run is True:
            self.val_dataloader = DataLoader(VideoDataset(
                self.config.val.val_list_file), batch_size=self.config.val.val_batch_size, num_workers=self.config.val.val_num_workers)

        # Test Data Loaders
        if self.config.test.test_run is True:
            self.test_dataloader = DataLoader(VideoDataset(
                self.config.test.test_list_file), batch_size=self.config.test.test_batch_size, num_workers=self.config.test.test_num_workers)

        # Model
        self.model = NaburangiV1().to(self.device)
        self.weight_init()

        # Train Preset
        if self.config.train.train_run is True:
            # Criterian
            self.criterion = nn.CrossEntropyLoss()

            # Optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.graph.optimizer.learning_rate, betas=(self.config.graph.optimizer.beta1, self.config.graph.optimizer.beta2))

            # Tensorboard
            if self.config.summary_writer_dir is not None:
                if os.path.exists(self.config.summary_writer_dir) is False:
                    os.makedirs(self.config.summary_writer_dir)
                self.summary_writer = SummaryWriter(os.path.join(
                    self.config.summary_writer_dir, self.start_datetime))

            # Save Check point
            if self.config.save_checkpoint is not None:
                if os.path.exists(os.path.join(self.config.save_checkpoint, self.start_datetime)) is False:
                    os.makedirs(os.path.join(
                        self.config.save_checkpoint, self.start_datetime))

    def weight_init(self):
        if self.config.load_checkpoint is not None:
            self.load_checkpoint()
        else:
            if self.config.s3d_pretrained_weight is not None:
                s3d_pretrained = torch.load(self.config.s3d_pretrained_weight)
                if 'model_state_dict' in s3d_pretrained.keys():
                    self.model.s3d.load_state_dict(
                        s3d_pretrained['model_state_dict'])
            if self.config.face_s3d_pretrained_weight is not None:
                face_s3d_pretrained = torch.load(
                    self.config.face_s3d_pretrained_weight)
                if 'model_state_dict' in face_s3d_pretrained.keys():
                    self.model.face_s3d.load_state_dict(
                        face_s3d_pretrained['model_state_dict'])

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.load_checkpoint)
        if 'model_state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint.keys():
            self.current_epoch = checkpoint['epoch']
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.save_checkpoint, self.start_datetime, str(self.current_epoch)))

    def run(self):
        if self.config.train.train_run is True:
            self.train()

        if self.config.test.test_run is True:
            self.test()

    def train(self):
        self.model.train()
        for epoch in range(self.current_epoch, self.config.train.epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            if epoch % self.config.val.val_interval == 0 and self.config.val.val_run is True:
                self.validate()

    def train_one_epoch(self):
        train_count = 0
        train_loss_sum = 0
        train_accuracy_sum = 0
        for step, sample in enumerate(self.train_dataloader, start=1):
            # Read data
            vframes = sample['vframes'].to(self.device)
            face_vframes = sample['face_vframes'].to(self.device)
            aframes = sample['aframes'].to(self.device)
            label = torch.LongTensor(sample['label']).to(self.device)
            output = self.model(vframes, face_vframes,
                                aframes)  # (batch, class)

            loss = self.criterion(output, label)
            predict = torch.max(output, dim=-1)[1]

            # Metrics
            train_loss_sum += loss.item() * output.size(0)
            train_accuracy_sum += (predict == label).sum().item()
            train_count += output.size(0)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if step % 10 == 0:
                print('{} EPOCH {} / {} - Loss: {:.6f}, Accuracy: {:.4f}%'.format(self.current_epoch, step,
                                                                                  len(self.train_dataloader), train_loss_sum / train_count, train_accuracy_sum / train_count * 100))
        # End of Epoch
        if self.config.save_checkpoint is not None:
            self.save_checkpoint()

        # Summary
        if self.config.summary_writer_dir is not None:
            self.summary_writer.add_scalar(
                'loss/train', train_loss_sum / train_count, self.current_epoch)
            self.summary_writer.add_scalar(
                'accuracy/train', train_accuracy_sum / train_count, self.current_epoch)
            if self.current_epoch == 1:
                with torch.no_grad():
                    self.summary_writer.add_graph(
                        self.model, (vframes, face_vframes, aframes))

        print('{} EPOCH - Loss: {:.6f}, Accuracy: {:.4f}%'.format(self.current_epoch,
                                                                  train_loss_sum / train_count, train_accuracy_sum / train_count * 100))

    def evaluate(self):
        self.model.eval()
        val_count = 0
        val_accuracy_sum = 0
        for sample in self.val_loader:
            vframes = sample['vframes'].to(self.device)
            face_vframes = sample['face_vframes'].to(self.device)
            aframes = sample['aframes'].to(self.device)
            label = torch.LongTensor(sample['label']).to(self.device)
            output = self.model(vframes, face_vframes, aframes)

            predict = torch.max(output, dim=-1)[1]
            val_count = output.size(0)
            val_accuracy_sum += (predict, label).sum().item()

        # Summary
        if self.config.summary_writer_dir is not None:
            self.summary_writer.add_scalar(
                'accuracy/val', val_accuracy_sum / val_count, self.current_epoch)

        print('Validation {} EPOCH - Accuracy: {:.4f}%'.format(self.current_epoch,
                                                               val_accuracy_sum / val_count * 100))

    def test(self):
        pass

    def finalize(self):
        pass
