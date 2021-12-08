#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import math

import torch
from torch._C import Size, wait
import torch.backends.cudnn
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

CLASS_COUNT = 15

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=None, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=64,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=5,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument("--data-aug-hflip", action="store_true")
parser.add_argument("--data-aug-brightness", type=float, default=0)
parser.add_argument("--dropout", type=float, default=0)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=args.data_aug_brightness),
            transforms.ToTensor(),
        ]
    )
    if args.data_aug_hflip:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transform])
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    # train_dataset = torchvision.datasets.CIFAR10(
    #     args.dataset_root, train=True, download=True, transform=transform
    # )
    full_dataset = DCASE("ADL_DCASE_DATA/development", clip_duration=3)
    dataset = DCASE("ADL_DCASE_DATA/development", clip_duration=3)
    eval_dataset = DCASE("ADL_DCASE_DATA/evaluation", clip_duration=3)
    # test_dataset = torchvision.datasets.CIFAR10(
    #     args.dataset_root, train=False, download=False, transform=transforms.ToTensor()
    # )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (math.floor(len(dataset) * 0.7), math.ceil(len(dataset) * 0.3))
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    model = CNN(class_count=CLASS_COUNT, dropout=0.01)

    criterion = nn.CrossEntropyLoss()

    if args.learning_rate is None:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    non_full_trainer = NonFullTrainer(
        model,
        train_loader,
        test_loader,
        dataset.num_clips,
        criterion,
        optimizer,
        summary_writer,
        DEVICE,
    )
    non_full_trainer.train(
        250,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    full_dataset_loader = torch.utils.data.DataLoader(
        full_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    full_trainer = FullTrainer(
        model,
        full_dataset_loader,
        eval_loader,
        dataset.num_clips,
        criterion,
        optimizer,
        summary_writer,
        DEVICE,
    )
    full_trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    torch.save(model.state_dict(), "output_weights")
    summary_writer.close()


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
        self._num_clips = self._total_duration // self._clip_duration

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
    def __init__(self, class_count: int, dropout: float):
        super().__init__()
        self.class_count = class_count

        # self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.AdaptiveMaxPool2d((12, 1))

        # 1 layer
        self.fc1 = nn.Linear(3072, self.class_count)
        self.initialise_layer(self.fc1)

        # 2 layers
        # self.fc1 = nn.Linear(3072, 2000)
        # self.initialise_layer(self.fc1)
        # self.fc2 = nn.Linear(2000, self.class_count)
        # self.initialise_layer(self.fc1)

        self.batchNorm1 = nn.BatchNorm2d(128)
        self.batchNorm2 = nn.BatchNorm2d(256)

    def forward(self, input_spectrograms: torch.Tensor) -> torch.Tensor:
        # TODO swap atchNorm and relu order
        x = F.relu(self.conv1(input_spectrograms))
        x = self.batchNorm1(x)
        # 128 x 60 x 150
        x = self.pool1(x)
        # 128 x 60 x 150
        x = F.relu(self.conv2(x))
        x = self.batchNorm2(x)
        # print(x.shape)
        # print("Pool 2")
        x = self.pool2(x)
        # print("Flatten")
        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.softmax(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        number_of_clips: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.number_of_clips = number_of_clips
        self.step = 0
        self.training_name = ""

    def print_metrics(
        self,
        epoch,
        accuracy,
        loss,
        data_load_time,
        step_time,
    ):
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

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar(f"{self.training_name}_epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            f"{self.training_name}_accuracy", {"train": accuracy}, self.step
        )
        self.summary_writer.add_scalars(
            f"{self.training_name}_loss", {"train": float(loss.item())}, self.step
        )
        self.summary_writer.add_scalar(
            f"{self.training_name}_time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
            f"{self.training_name}_time/data", step_time, self.step
        )

    def validate(self) -> float:
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                segments = batch.view((-1, 1, 60, 150))  # TODO put this in function
                logits = self.model(segments)
                logits = torch.reshape(
                    logits,
                    (
                        -1,
                        self.number_of_clips,
                        self.model.class_count,
                    ),
                )
                logits = logits.mean(1)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
            f"{self.training_name}_accuracy", {"test": accuracy}, self.step
        )
        for key, value in all_classes_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        ).items():
            self.summary_writer.add_scalars(
                f"{self.training_name}_class_accuracy_{key}", {"test": value}, self.step
            )
        self.summary_writer.add_scalars(
            f"{self.training_name}_loss", {"test": average_loss}, self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        print(
            f"All classes accuracy: {all_classes_accuracy(np.array(results['labels']), np.array(results['preds'])).items()}"
        )

        return accuracy


class NonFullTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_name = "non_full"

    def train(
        self,
        epochs: int,
        val_frequency: int = 5,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
    ):
        self.model.train()

        current_weights = self.model.state_dict()
        current_accuracy = 0
        current_best_accuracy = 0
        epochs_since_improvement = 0

        for epoch in range(start_epoch, epochs):
            if epochs_since_improvement > 100:
                break
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                segments = batch.view((-1, 1, 60, 150))
                logits = self.model.forward(segments)

                logits = torch.reshape(
                    logits,
                    (
                        -1,
                        self.number_of_clips,
                        self.model.class_count,
                    ),
                )
                logits = logits.mean(1)
                # for index, item in enumerate(logits[0]):
                # print(f"My prediction for class {index} is: {item}")
                # print(logits.shape)

                loss = self.criterion(logits, labels)
                # print(labels)
                # print(labels.shape)
                # print(loss)
                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(
                        epoch,
                        accuracy,
                        loss,
                        data_load_time,
                        step_time,
                    )
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar(
                f"{self.training_name}_epoch", epoch, self.step
            )
            if ((epoch + 1) % val_frequency) == 0:
                accuracy = self.validate()
                if accuracy > current_accuracy:
                    print("Improvement, weights saved")
                    current_weights = self.model.state_dict()
                    current_accuracy = accuracy
                else:
                    print("No imporvement, using old weights")
                    self.model.load_state_dict(current_weights)
                    if epoch >= 100:
                        print("No improvement and 100 epochs done, breaking")
                        break

                if current_accuracy <= current_best_accuracy:
                    print(
                        f"No overall imporvement over current best {current_best_accuracy}, epoch since last imporvement: {epochs_since_improvement}"
                    )
                    epochs_since_improvement += val_frequency
                else:
                    current_best_accuracy = current_accuracy
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()


class FullTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_name = "full"

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
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

                segments = batch.view((-1, 1, 60, 150))
                logits = self.model.forward(segments)

                logits = torch.reshape(
                    logits,
                    (
                        -1,
                        self.number_of_clips,
                        self.model.class_count,
                    ),
                )
                logits = logits.mean(1)
                # for index, item in enumerate(logits[0]):
                # print(f"My prediction for class {index} is: {item}")
                # print(logits.shape)

                loss = self.criterion(logits, labels)
                # print(labels)
                # print(labels.shape)
                # print(loss)
                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar(
                f"{self.training_name}_epoch", epoch, self.step
            )
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def class_accuracy(
    class_number: int,
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
):
    class_mask = torch.Tensor(len(preds))
    class_mask.fill_(class_number)
    assert len(labels) == len(preds)
    assert len(labels) == len(class_mask)
    preds = torch.Tensor(preds)
    labels = torch.Tensor(labels)

    actual_number_of_class = (labels == class_mask).sum()
    if actual_number_of_class == 0:
        return float(-1)
    return float(((labels == preds) & (labels == class_mask)).sum()) / float(
        actual_number_of_class
    )


def all_classes_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
):
    print("Get all class accuracies")
    return {
        class_number: class_accuracy(class_number, labels, preds)
        for class_number in range(CLASS_COUNT)
    }


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"dropout={args.dropout}_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"momentum=0.9_"
        + f"brightness={args.data_aug_brightness}_"
        + ("hflip_" if args.data_aug_hflip else "")
        + f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
