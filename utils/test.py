import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
from torchmetrics.classification import F1Score
from torchvision import datasets, transforms

from dataloader import (
    english_test_loader,
    english_train_loader,
    mandarin_test_loader,
    mandarin_train_loader,
)
from model import DANNModel
from utils.metrics import log_tensorboard, update_metrics

random.seed(42)
torch.manual_seed(42)

loaders_ = {
    "english_train": english_train_loader,
    "english_test": english_test_loader,
    "mandarin_train": mandarin_train_loader,
    "mandarin_test": mandarin_test_loader,
}


def test(args, dataset_name, metrics, writer, tag, epoch):
    dataloader = loaders_[dataset_name]
    data_name = dataset_name.split("_")[0]

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True
    batch_size = 128
    image_size = 224
    alpha = 0

    """ test """
    net = torch.load(
        os.path.join(args.model_path, f"{source}_{target}_model_epoch_{epoch}.pth")
    )
    net = net.eval()
    f1 = F1Score(task="multiclass", num_classes=31)

    loss_class = torch.nn.NLLLoss()

    if device == "cuda":
        net = net.to(device)
        loss_class = loss_class.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    f1_running = 0
    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if device == "cuda":
            t_img = t_img.to(device).to(torch.float32)
            t_label = t_label.to(device)

        class_output, _ = net(input=t_img, alpha=alpha)
        err_t_label = loss_class(class_output, t_label)

        pred = torch.argmax(class_output, dim=1)
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        # update val metrics
        metrics = update_metrics(metrics, pred, t_label)
        writer.add_scalar(f"Loss/class/{tag}/val", err_t_label, epoch)

        f1_running += f1(pred.cpu(), t_label.data.view_as(pred).cpu())

        i += 1

    metrics = log_tensorboard(
        writer, f"class/{tag}/val", metrics, epoch, source, target
    )

    accu = n_correct.data.numpy() * 1.0 / n_total
    f1_running /= n_total

    return accu, f1_running, metrics
