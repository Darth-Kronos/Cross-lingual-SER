import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import psutil
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import (
    english_test_loader,
    english_train_loader,
    mandarin_test_loader,
    mandarin_train_loader,
)
from model import DANNModel
from utils.metrics import log_tensorboard, set_metrics, update_metrics
from utils.parse_args import parse_args
from utils.perturb import perturb_loader
from utils.test import test

random.seed(42)
torch.manual_seed(42)

loaders_ = {
    "english_train": english_train_loader,
    "english_test": english_test_loader,
    "mandarin_train": mandarin_train_loader,
    "mandarin_test": mandarin_test_loader,
}


def main(args):
    dataloader_source = loaders_[args.source_dataset]
    dataloader_target = loaders_[args.target_dataset]
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    if args.perturb:
        print("Perturbing training dataloaders")

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # load model
    net = DANNModel()

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len_dataloader,
        epochs=args.epochs,
        anneal_strategy="cos",
    )

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if device == "cuda":
        net = net.to(device)
        loss_class = loss_class.to(device)
        loss_domain = loss_domain.to(device)

    for p in net.parameters():
        p.requires_grad = True

    if args.perturb:
        log_dir = (
            f"{args.tensorboard_log_dir}/{source}_{target}_{args.encoder}_perturbed"
        )
    else:
        log_dir = f"{args.tensorboard_log_dir}/{source}_{target}_{args.encoder}"
    writer = SummaryWriter(log_dir=log_dir)  # create a TensorBoard writer
    d_t_train_metrics, d_t_val_metrics = set_metrics(device, num_classes=2)
    c_s_train_metrics, c_s_val_metrics = set_metrics(device, num_classes=5)
    c_t_train_metrics, c_t_val_metrics = set_metrics(device, num_classes=5)

    # training
    results = []
    best_accu_t = 0.0
    for epoch in range(args.epochs):

        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = next(data_source_iter)

            if args.perturb:
                s_img, s_label = perturb_loader(data_source)
            else:
                s_img, s_label = data_source

            net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if device == "cuda":
                s_img = s_img.to(device).to(torch.float32)
                s_label = s_label.to(device)
                domain_label = domain_label.to(device)

            class_output, domain_output = net(input=s_img, alpha=alpha)
            class_s_pred = torch.argmax(class_output, dim=1)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            if args.perturb:
                t_img, t_label = perturb_loader(data_target)
            else:
                t_img, t_label = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if device == "cuda":
                t_img = t_img.to(device).to(torch.float32)
                t_label = t_label.to(device)
                domain_label = domain_label.to(device)

            class_t_output, domain_output = net(input=t_img, alpha=alpha)
            domain_t_pred = torch.argmax(domain_output, dim=1)
            class_t_pred = torch.argmax(class_t_output, dim=1)
            err_t_label = loss_class(class_t_output, t_label)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            # update metrics
            d_t_train_metrics = update_metrics(
                d_t_train_metrics, domain_t_pred, domain_label
            )
            c_s_train_metrics = update_metrics(c_s_train_metrics, class_s_pred, s_label)
            c_t_train_metrics = update_metrics(c_t_train_metrics, class_t_pred, t_label)
            writer.add_scalar(f"Loss/domain/target/train", err_t_domain, epoch)
            writer.add_scalar(f"Loss/class/target/train", err_t_label, epoch)
            writer.add_scalar(f"Loss/class/source/train", err_s_label, epoch)
            writer.add_scalar(f"Loss/overall/train", err, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

            scheduler.step()

            sys.stdout.write(
                "\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f"
                % (
                    epoch,
                    i + 1,
                    len_dataloader,
                    err_s_label.data.detach().cpu().numpy(),
                    err_s_domain.data.detach().cpu().numpy(),
                    err_t_domain.data.detach().cpu().item(),
                )
            )
            sys.stdout.flush()
            torch.save(
                net,
                f"{args.model_path}/{source}_{target}_model_epoch_{epoch}.pth",
            )

        d_t_train_metrics = log_tensorboard(
            writer, "domain/target/train", d_t_train_metrics, epoch, source, target
        )
        c_s_train_metrics = log_tensorboard(
            writer, "class/source/train", c_s_train_metrics, epoch, source, target
        )
        c_t_train_metrics = log_tensorboard(
            writer, "class/target/train", c_t_train_metrics, epoch, source, target
        )

        # Log RAM usage
        mem_info = psutil.virtual_memory()
        ram_usage = mem_info.used / (1024**3)  # RAM usage in GB
        writer.add_scalar("RAM usage", ram_usage, epoch)

        # Log model parameters
        #for name, param in net.named_parameters():
        #    writer.add_histogram(
        #        f"Parameters/{name}", param.clone().cpu().data.numpy(), epoch
        #    )

        print("\n")
        accu_s, f1_s, c_s_val_metrics = test(
            args, args.source_dataset, c_s_val_metrics, writer, "source", epoch
        )
        print("Accuracy of the %s dataset: %f" % (source, accu_s))
        accu_t, f1_t, c_t_val_metrics = test(
            args, args.target_dataset, c_t_val_metrics, writer, "target", epoch
        )
        print("Accuracy of the %s dataset: %f\n" % (target, accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(net, f"{args.model_path}/{source}_{target}_model_epoch_best.pth")
        results.append(
            [
                epoch,
                err_s_label.data.detach().cpu().numpy(),
                err_s_domain.data.detach().cpu().numpy(),
                err_t_domain.data.detach().cpu().item(),
                accu_s,
                f1_s,
                accu_t,
                f1_t,
            ]
        )

        if epoch != (args.epochs - 1):
            os.remove(f"{args.model_path}/{source}_{target}_model_epoch_{epoch}.pth")

    print("============ Summary ============= \n")
    print("Accuracy of the %s dataset: %f" % (source, best_accu_s))
    print("Accuracy of the %s dataset: %f" % (target, best_accu_t))
    print(
        "Corresponding model was save in "
        + args.model_path
        + f"/{source}_{target}_model_epoch_best.pth"
    )

    writer.close()


if __name__ == "__main__":
    args = parse_args()

    main(args)
