from pathlib import Path
from typing import NamedTuple
import time
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.optimizer import Optimizer

import click
import pandas as pd
import numpy as np


class SpectrogramShape(NamedTuple):
    height: int
    duration: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@click.command()
@click.option("--learning_rate", default=1e-1)
@click.option("--batch_size", default=64)
def main(learning_rate, batch_size):
    train_dataset = DCASE("ADL_DCASE_DATA/development", clip_duration=3)
    model = CNN()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count(),
    )
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, criterion, optimizer, DEVICE)
    trainer.train(50)


class DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv(
            (self._root_dir / "labels.csv"), names=["file", "label"]
        )
        self._labels["label"] = self._labels.label.astype("category").cat.codes.astype(
            "int"
        )  # create categorical labels
        self._clip_duration = clip_duration
        self._total_duration = 30  # DCASE audio length is 30s

        self._data_len = len(self._labels)

    @property
    def spectrogram_shape(self):
        return SpectrogramShape(height=60, duration=self._clip_duration)

    def __getitem__(self, index):
        # reading spectrograms
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / "audio" / filename
        spec = torch.from_numpy(np.load(filepath))

        # splitting spec
        spec = self._trim(spec)
        return spec, label

    def _trim(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            # spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    @property
    def num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self._initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(5, 5), padding=(2, 2)
        )
        self._initialise_layer(self.conv2)

        # TODO change this!
        self.pool2 = nn.AdaptiveMaxPool2d((4, 1))
        # self.pool2 = nn.MaxPool2d(kernel_size=(3, 30), stride=(3, 30))

        # 1024
        self.fc1 = nn.Linear(1024, 15)
        self._initialise_layer(self.fc1)

    def forward(self, input_spectrograms: torch.Tensor):
        x = F.relu(self.conv1(input_spectrograms))
        # 128 x 60 x 150
        x = self.pool1(x)
        # 128 x 60 x 150
        x = F.relu(self.conv2(x))
        # print(x.shape)
        # print("Pool 2")
        x = self.pool2(x)
        # print(x.shape)
        # print("Flatten")
        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        return x

    @staticmethod
    def _initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.step = 0

    def train(
        self,
        epochs: int,
        print_frequency: int = 1,
        # log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # Step
                print(f"{batch=}")
                print(f"{batch.shape=}")
                segments = torch.flatten(batch, end_dim=1)
                segments = segments[:, None, :]
                logits = self.model.forward(segments)
                print(f"{segments=}")
                print(f"{segments.shape=}")
                # print(f"{logits=}")

                # Average segments for each clip
                logits = torch.reshape(logits, (-1, 10, 15))
                # print(f"Before mean {logits=}")
                logits = logits.mean(1)
                # print(f"{logits=}")
                # print(f"{labels=}")

                # import sys
                # sys.exit(0)

                # Compute loss
                print("Helllo!")
                print(f"{logits=}")
                print(f"{logits.shape=}")
                print(f"{labels=}")
                loss = self.criterion(logits, labels)
                loss.backward()

                # Optimize
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Compute accuracy
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = self.compute_accuracy(labels, preds)

                # Log it
                # if (self.step + 1) % log_frequency == 0:
                #     self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if (self.step + 1) % print_frequency == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    @staticmethod
    def compute_accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
        """
        Args:
            labels: ``(batch_size, class_count)`` tensor or array containing example labels
            preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        """
        assert len(labels) == len(preds)
        return float((labels == preds).sum()) / len(labels)


if __name__ == "__main__":
    main()
