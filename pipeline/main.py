import config
import utils
from dataset import TrainDataset, TestDataset
from augment import get_transforms
from model import CustomEfficientNet
from train import train_fn
from valid import valid_fn

# Regular Imports
import pandas as pd
from sklearn import model_selection
import time

# Torch Imports
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

OUTPUT_DIR = "/"
train = pd.read_csv("pipeline/data/train.csv")

LOGGER = utils.init_logger()
utils.seed_torch(config.seed)

# Creating CV Strategy
Fold = model_selection.StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[config.target_col])):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)


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
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if config.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=config.factor, patience=config.patience, verbose=True, eps=config.eps
            )
        elif config.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.min_lr, last_epoch=-1)
        elif config.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=config.T_0, T_mult=1, eta_min=config.min_lr, last_epoch=-1
            )
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomEfficientNet(config.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex
    # ====================================================
    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    def get_criterion():
        if config.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif config.criterion == "LabelSmoothing":
            criterion = LabelSmoothingLoss(classes=config.target_size, smoothing=config.smoothing)
        elif config.criterion == "FocalLoss":
            criterion = FocalLoss().to(device)
        elif config.criterion == "FocalCosineLoss":
            criterion = FocalCosineLoss()
        elif config.criterion == "SymmetricCrossEntropyLoss":
            criterion = SymmetricCrossEntropy().to(device)
        elif config.criterion == "BiTemperedLoss":
            criterion = BiTemperedLogisticLoss(t1=config.t1, t2=config.t2, smoothing=config.smoothing)
        elif config.criterion == "TaylorCrossEntropyLoss":
            criterion = TaylorCrossEntropyLoss(smoothing=config.smoothing)
        return criterion

    # ====================================================
    # loop
    # ====================================================
    criterion = get_criterion()
    LOGGER.info(f"Criterion: {criterion}")

    best_score = 0.0
    best_loss = np.inf

    for epoch in range(config.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[config.target_col].values

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
                {"model": model.state_dict(), "preds": preds}, OUTPUT_DIR + f"{config.model_name}_fold{fold}_best.pth"
            )

    check_point = torch.load(OUTPUT_DIR + f"{config.model_name}_fold{fold}_best.pth")
    valid_folds[[str(c) for c in range(5)]] = check_point["preds"]
    valid_folds["preds"] = check_point["preds"].argmax(1)

    return valid_folds


def main():

    """
    Prepare: 1.train  2.test  3.submission  4.folds
    """

    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df[config.target_col].values
        score = get_score(labels, preds)
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
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)


if __name__ == "__main__":
    main()
