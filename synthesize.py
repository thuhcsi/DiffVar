import os
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FastSpeech2, DiffVariancePredictor
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, synth_samples
from dataset import Dataset

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthesize_batch(batch, model, configs, vocoder, outdir, control_values,
                     use_gt_var:bool=False, var_diff_pred=None):
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    (
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max_text_lens,
        mels,
        mel_lens,
        max_mel_lens,
        pitches,
        energies,
        durations,
        quasi_flags,
    ) = batch

    if var_diff_pred is not None:
        durations = torch.clamp(
            (torch.round(torch.exp(var_diff_pred[:, :, 2]))),
            min=0,
        )
        for i, (l, f) in enumerate(zip(text_lens, quasi_flags)):
            durations[i, l-sum(f):] = 0.
    elif not use_gt_var:
        durations = None

    batch = (
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max(text_lens),
        None,
        mel_lens if use_gt_var else None,
        max_mel_lens if use_gt_var else None,
        var_diff_pred[:, :, 0] if var_diff_pred is not None else \
            (pitches if use_gt_var else None),
        var_diff_pred[:, :, 1] if var_diff_pred is not None else \
            (energies if use_gt_var else None),
        durations,
        quasi_flags,
    )

    # Forward
    pitch_control, energy_control, duration_control = control_values
    output = model(
        *(batch[2:-1]),
        quasi_symbols = batch[-1],
        p_control=pitch_control,
        e_control=energy_control,
        d_control=duration_control
    )
    
    preprocess_config, model_config, train_config = configs
    synth_samples(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
        outdir,
    )

def synthesize(fs2_model, configs, vocoder, dataloader, outdir, diffvar_model=None,
               use_gt_var=False, control_values=[1.,1.,1.], control_dv_spker=None,control_dec_spkers=None):
    for batchs in tqdm(dataloader):
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                if diffvar_model is not None:
                    if control_dv_spker is not None:
                        batch[2].fill_(control_dv_spker)
                    var, cond, var_mask = diffvar_model.validation_step(batch[2:], fs2_model)
                    # pitch_diff_pred = var[:, :, 0]
                else:
                    # pitch_diff_pred = None
                    # var = None
                    if control_dv_spker is not None:
                        batch[2].fill_(control_dv_spker)
                    outputs = fs2_model(*(list(batch[2:-4])+[None]*3+[batch[-1]]), skip_decoder=True)
                    var = torch.stack(outputs[2:5], dim=-1)

                
                if control_dec_spkers is not None:
                    for control_dec_spker in control_dec_spkers:
                        batch[2].fill_(control_dec_spker)
                        synthesize_batch(
                            batch, fs2_model, configs, vocoder,
                            os.path.join(outdir, str(control_dec_spker)), control_values,
                            use_gt_var=use_gt_var, var_diff_pred=var)

def getConfig(config_name):
    preprocess_config = yaml.load(
        open(f"config/{config_name}/preprocess.yaml", "r"),
        Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(f"config/{config_name}/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(f"config/{config_name}/train.yaml", "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config

if __name__ == "__main__":
    # Read Config
    preprocess_config, model_config, train_config = configs = getConfig("zjl_enc_detach")

    # Get model
    fs2_model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt_path = os.path.join(
        train_config["path"]["ckpt_path"],
        "900000.pth.tar",
    )
    ckpt = torch.load(ckpt_path)
    fs2_model.load_state_dict(ckpt["model"])
    fs2_model.eval()
    fs2_model.requires_grad_ = False

    preprocess_config, model_config, train_config = configs = getConfig("zdl2_split")
    diffvar_model = DiffVariancePredictor(model_config).to(device)
    ckpt_path = os.path.join(
        train_config["path"]["ckpt_path"],
        "900000.pth.tar",
    )
    ckpt = torch.load(ckpt_path)
    diffvar_model.load_state_dict(ckpt["model"])
    diffvar_model.eval()
    diffvar_model.requires_grad_ = False
    print(get_param_num(diffvar_model))

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Get dataset
    dataset = Dataset(
        "val.txt",
        preprocess_config,
        train_config,
        sort=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dataset.collate_fn,
    )

    synthesize(
        fs2_model, configs, vocoder, dataloader,
        f"output/result",
        diffvar_model=diffvar_model,
        use_gt_var=False,
    )