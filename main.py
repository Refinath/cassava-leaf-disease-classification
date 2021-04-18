import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import utils
from sklearn import model_selection
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from cassava import config, loss
from cassava.augment import get_transforms
from cassava.dataset import TestDataset, TrainDataset
from cassava.model import CassavaClassifier
from cassava.train import train_fn
from cassava.valid import valid_fn

# # Initializations
OUTPUT_DIR = "/"
train = pd.read_csv("data/train.csv")

LOGGER = utils.init_logger()
utils.seed_torch(config.SEED)

# Creating CV Strategy
folds = train.copy()
fold_strategy = model_selection.StratifiedKFold(
    n_splits=config.N_FOLD, shuffle=True, random_state=config.SEED
)
for n, (train_index, val_index) in enumerate(fold_strategy.split(folds, folds[config.TARGET_COL])):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")


def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data="train"))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if config.SCHEDULER == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.factor,
                patience=config.patience,
                verbose=True,
                eps=config.eps,
            )
        elif config.SCHEDULER == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=config.T_max, eta_min=config.MIN_LR, last_epoch=-1
            )
        elif config.SCHEDULER == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=config.T_0, T_mult=1, eta_min=config.MIN_LR, last_epoch=-1
            )
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CassavaClassifier(config.MODEL_NAME, pretrained=True)
    model.to(device)

    optimizer = Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY, amsgrad=False
    )
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex
    # ====================================================
    if config.APEX:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    def get_criterion():
        if config.CRITERION == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif config.CRITERION == "FocalCosineLoss":
            criterion = loss.FocalCosineLoss()
        elif config.CRITERION == "BiTemperedLoss":
            criterion = loss.BiTemperedLogisticLoss(
                t1=config.t1, t2=config.t2, smoothing=config.smoothing
            )
        return criterion

    # ====================================================
    # loop
    # ====================================================
    criterion = get_criterion()
    LOGGER.info(f"Criterion: {criterion}")

    best_score = 0.0

    for epoch in range(config.EPOCHS):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[config.TARGET_COL].values

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = utils.get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Accuracy: {score}")

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"{config.MODEL_NAME}_fold{fold}_best.pth",
            )

    check_point = torch.load(OUTPUT_DIR + f"{config.MODEL_NAME}_fold{fold}_best.pth")
    valid_folds[[str(c) for c in range(5)]] = check_point["preds"]
    valid_folds["preds"] = check_point["preds"].argmax(1)

    return valid_folds


def main():

    """
    Prepare: 1.train  2.test  3.submission  4.folds
    """

    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df[config.TARGET_COL].values
        score = utils.get_score(labels, preds)
        LOGGER.info(f"Score: {score:<.5f}")

    if config.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(config.n_fold):
            if fold in config.trn_fold:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info("========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)


if __name__ == "__main__":
    main()
