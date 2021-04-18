import time

import numpy as np
import torch
from utils import AverageMeter, cutmix, time_since

from cassava import config


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device).float()
        labels = labels.to(device).long()
        batch_size = labels.size(0)
        # Cut Mix
        mix_decision = np.random.rand()
        if mix_decision < 0.25:
            images, labels = cutmix(images, labels, 1.0)

        y_preds = model(images.float())
        if mix_decision < 0.50:
            loss = criterion(y_preds, labels[0]) * labels[2] + criterion(y_preds, labels[1]) * (
                1.0 - labels[2]
            )
        else:
            loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if config.GRADIENT_ACCUM_STEPS > 1:
            loss = loss / config.GRADIENT_ACCUM_STEPS
        if config.APEX:
            from apex import amp

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        if (step + 1) % config.GRADIENT_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.PRINT_FREQ == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{0}][{1}/{2}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                # 'LR: {lr:.6f}  '
                .format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=time_since(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                    # lr=scheduler.get_lr()[0],
                )
            )
    return losses.avg
