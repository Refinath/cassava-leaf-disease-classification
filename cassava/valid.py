import time

import numpy as np
import torch
from utils import AverageMeter, time_since

from cassava import config


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to("cpu").numpy())
        if config.GRADIENT_ACCUM_STEPS > 1:
            loss = loss / config.GRADIENT_ACCUM_STEPS
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.PRINT_FREQ == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{0}/{1}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=time_since(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions
