from pathlib import Path
from typing import NamedTuple
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.optimizer import Optimizer
import torchaudio

import click
import pandas as pd


class SpectrogramShape(NamedTuple):
    height: int
    duration: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@click.command()
@click.option("--learning_rate")
def main(count):
    train_dataset = DCASE("ADL_DCASE_DATA/development", clip_duration=10)
    model = CNN()

    train_loader = torch.utils.data.DataLoader(data_loader)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, criterion, optimizer, DEVICE)


class DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv(
            (self._root_dir / "labels.csv"), names=["file", "label"]
        )
        self._labels["label"] = self._labels.label.astype("category").cat.codes.astype(
            "int"
        )  # create categorical labels
        self._clip_duration: int = clip_duration
        self._total_duration = 30  # DCASE audio length is 30s

        self._sample_rate = 44100  # DCASE sampling rate is 44100

        # creating melspec function
        win_size = int(round(40 * self._sample_rate / 1e3))  # 40ms window length
        self._spec_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            n_fft=win_size,
            n_mels=60,
            hop_length=win_size // 2,  # 50% overlapping windows
            window_fn=torch.hamming_window,
            power=2,
            # normalized=True, This could be a touch rash
        )

        self._data_len = len(self._labels)

    @property
    def spectrogram_shape(self):
        return SpectrogramShape(height=60, duration=self._clip_duration)

    def __getitem__(self, index):
        # reading raw audio
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / "audio" / filename
        data_array, sample_rate = torchaudio.load(filepath)

        # make sure using correct sampling rate
        assert sample_rate == self._sample_rate, (
            "Sample rate doesn't match expected rate. "
            "Can not create spectrogram as intended, likely an issue with data."
        )
        # creating spectrogram and splitting
        spec = self._make_spec(data_array)
        spec = self._trim(spec)
        return spec, label

    def _make_spec(self, data_array: torch.Tensor) -> torch.Tensor:
        """
        Create spectrogram using data input
        :param data_array: tensor containing raw audio of shape [1, 1323001 (sample_rate * audio length(30s))]
        :return: tensor containing log mel spectrogram of shape [1, 60, 1501]
        """
        spec = self._spec_fn(data_array).log2()
        return spec

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
            spec_clip = spec[:, :, start:end]
            spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        sequences = torch.stack(all_clips)
        return sequences

    def get_num_clips(self) -> int:
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
            out_channels=1,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))

        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(5, 5), padding=(2, 2)
        )

    def forward(self, input_spectrograms: torch.Tensor):
        x = F.relu(self.conv1(input_spectrograms))
        x = self.pool1(x)
        return x


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
        print_frequency: int = 20,
        # log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                # Step
                logits = self.model.forward(batch)

                # Compute loss
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
