import argparse
import os
from turtle import st

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_vocoder, get_param_num, ScheduledOptim
from utils.tools import to_device
from model import FastSpeech2, DiffVariancePredictor
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # Prepare model
    fs2_model = FastSpeech2(preprocess_config, model_config)
    fs2_model.to(device)
    pre_trained_fs2_model = torch.load(args.fs_path)
    fs2_model.load_state_dict(pre_trained_fs2_model["model"],strict=True)

    model = DiffVariancePredictor(model_config)
    model.to(device)
    
    optimizers = [ScheduledOptim(
        d_fn, train_config, model_config, 0
    ) for d_fn in model.denoise_fn]
    
    print("Number of FS2 Total Parameters:", get_param_num(fs2_model))
    print("Number of FS2 pitch predictor:", get_param_num(fs2_model.variance_adaptor.pitch_predictor))
    print("Number of FS2 energy predictor:", get_param_num(fs2_model.variance_adaptor.energy_predictor))
    print("Number of FS2 duration predictor:", get_param_num(fs2_model.variance_adaptor.duration_predictor))
    print("Number of DiffVar predictor:", get_param_num(model))

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                losses = model.training_step(batch[2:], fs2_model)

                # Cal Loss & Backward
                for ch_loss in losses:
                    ch_loss = ch_loss / grad_acc_step
                    ch_loss.backward()

                if step % grad_acc_step == 0:

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    for optimizer in optimizers:
                        new_lr = optimizer.step_and_update_lr()
                        optimizer.zero_grad()
                    
                        

                if step % log_step == 0:
                    message1 = "Step {}/{}, ".format(step, total_step)
                    
                    loss_msgs, losses = [], [l.item() for l in losses]
                    for l, idx in zip(losses, model.in_dims):
                        sidx = f"Total Loss{str[idx]}"
                        loss_msgs.append(f"{sidx}: {l:.4f}")
                        train_logger.add_scalar(f"Loss/{sidx}", l, step)
                    
                    total_msg = message1 + ' '.join(loss_msgs)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(total_msg + "\n")
                    outer_bar.write(total_msg)

                    train_logger.add_scalar("Learning_rate", new_lr, step)

                if step % val_step == 0:
                    model.eval()
                    
                    mse_loss = nn.MSELoss()
                    p_loss, e_loss, log_d_loss = 0., 0., 0.
                    p_preds, e_preds, log_d_preds = [], [], []
                    p_tgts, e_tgts, log_d_tgts = [], [], []
                    for v_batchs in val_loader:
                        for v_batch in v_batchs:
                            v_batch = to_device(v_batch, device)
                            var, cond, var_mask = model.validation_step(v_batch[2:], fs2_model)
                            var_mask = ~var_mask
                            p_targets = v_batch[-4].masked_select(var_mask)
                            e_targets = v_batch[-3].masked_select(var_mask)
                            log_d_targets = torch.log(v_batch[-2].masked_select(var_mask))
                            p_pred = var[:, :, 0].masked_select(var_mask)
                            e_pred = var[:, :, 1].masked_select(var_mask)
                            log_d_pred = var[:, :, 2].masked_select(var_mask)
                            p_loss += mse_loss(p_pred, p_targets)# * len(v_batch[0])
                            e_loss += mse_loss(e_pred, e_targets)# * len(v_batch[0])
                            log_d_loss += mse_loss(log_d_pred, log_d_targets)# * len(v_batch[0])

                            p_preds.append(p_pred)
                            e_preds.append(e_pred)
                            log_d_preds.append(log_d_pred)
                            p_tgts.append(p_targets)
                            e_tgts.append(e_targets)
                            log_d_tgts.append(log_d_targets)
                            
                            break
                        break


                    val_logger.add_scalar("Loss/pitch_loss", p_loss, step)
                    val_logger.add_scalar("Loss/energy_loss", e_loss, step)
                    val_logger.add_scalar("Loss/duration_loss", log_d_loss, step)

                    p_preds = torch.cat(p_preds)
                    e_preds = torch.cat(e_preds)
                    log_d_preds = torch.cat(log_d_preds)
                    p_tgts = torch.cat(p_tgts)
                    e_tgts = torch.cat(e_tgts)
                    log_d_tgts = torch.cat(log_d_tgts)
                    val_logger.add_histogram("Hist/pitch_prediction", p_preds, step)
                    val_logger.add_histogram("Hist/energy_prediction", e_preds, step)
                    val_logger.add_histogram("Hist/duration_prediction", log_d_preds, step)
                    val_logger.add_histogram("Hist/pitch_targets", p_targets, step)
                    val_logger.add_histogram("Hist/energy_targets", e_tgts, step)
                    val_logger.add_histogram("Hist/duration_targets", log_d_tgts, step)
                    
                    for k, v in model.named_parameters():
                        val_logger.add_histogram(k, v, step)

                    message = f"Validation Step {step},Pitch loss: {p_loss:.4f}, Energy loss: {e_loss:.4f}, Duration loss: {log_d_loss:.4f}"

                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizers": [opt._optimizer.state_dict() for opt in optimizers], 
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-f", "--fs_path", type=str, required=True, help="path to pre-trained FastSpeech2 checkpoint"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
