import os

import mlconfig
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm
from tqdm import trange

from .models import EfficientNet


@mlconfig.register
class Trainer:
    def __init__(
        self,
        model: EfficientNet,
        optimizer: optim.Optimizer,
        train_loader: data.DataLoader,
        valid_loader: data.DataLoader,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        num_epochs: int,
        output_dir: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.num_classes = 6

        self.epoch = 1
        self.best_acc = 0

    def fit(self) -> None:
        epochs = trange(self.epoch, self.num_epochs + 1, desc="Epoch", ncols=0)
        for self.epoch in epochs:
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.validate()
            self.scheduler.step()

            self.save_checkpoint(os.path.join(self.output_dir, "checkpoint.pth"))
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc
                self.save_checkpoint(os.path.join(self.output_dir, "best.pth"))

            epochs.set_postfix_str(
                f"train loss: {train_loss}, train acc: {train_acc}, "
                f"valid loss: {valid_loss}, valid acc: {valid_acc}, "
                f"best valid acc: {self.best_acc:.2f}"
            )

    def train(self) -> tuple[float, float]:
        self.model.train()

        loss_metric = MeanMetric()
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes)

        train_loader = tqdm(self.train_loader, ncols=0, desc="Train")
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_metric.update(loss.item(), weight=x.size(0))
            acc_metric.update(output.cpu(), y.cpu())

            train_loader.set_postfix_str(
                f"train loss: {loss_metric.compute().item():.4f}, train acc: {acc_metric.compute().item():.4f}."
            )

        return loss_metric.compute().item(), acc_metric.compute().item()

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()

        loss_metric = MeanMetric()
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(device=self.device)

        valid_loader = tqdm(self.valid_loader, desc="Validate", ncols=0)
        print(self.device)
        for x, y in valid_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            loss_metric.update(loss.item(), weight=x.size(0))
            acc_metric.update(output, y)

            valid_loader.set_postfix_str(
                f"valid loss: {loss_metric.compute().float():.4f}, valid acc: {acc_metric.compute().item():.4f}."
            )

        return loss_metric.compute().float(), acc_metric.compute().item()

    def save_checkpoint(self, f: str) -> None:
        self.model.eval()

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_acc": self.best_acc,
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f: str) -> None:
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint["best_acc"]
