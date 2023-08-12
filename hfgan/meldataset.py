import math
import os
import random
from pathlib import Path

import json
import librosa
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

def get_dataset_filelist_DB6(a):
    with open(a.input_training_file, 'r', encoding='utf-8-sig') as f:
        training_files = [l.split("|")[2:4] for l in f.readlines()]
    with open(a.input_validation_file, 'r', encoding='utf-8-sig') as f:
        validation_files = [l.split("|")[2:4] for l in f.readlines()]
    return training_files, validation_files

def get_dataset_filelist_DBpara(a):
    with open(a.input_training_file, 'r', encoding='utf-8-sig') as f:
        training_files = [l.split("|")[0]+".npy" for l in f.readlines()]
    with open(a.input_validation_file, 'r', encoding='utf-8-sig') as f:
        validation_files = [l.split("|")[0]+".npy" for l in f.readlines()]
    return training_files, validation_files

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, audio_dir, mel_dir=None, split=False, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, win_center=False):
        
        self.audio_files = metadata
        self.audio_dir = Path(audio_dir)
        self.mel_dir = Path(mel_dir)
        
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss

        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.win_center = win_center

    def __getitem__(self, index):
        # modified Aug 25, 2021
        # audio_filepath, mel_filepath = self.audio_files[index]
        # audio_filepath = self.audio_dir / audio_filepath
        # mel_filepath = self.mel_dir / mel_filepath
        filepath = self.audio_files[index]
        audio_filepath = self.audio_dir / ("audio-"+filepath)
        if not audio_filepath.is_file():
            audio_filepath = self.audio_dir / filepath
        mel_filepath = self.mel_dir / ("mel-"+filepath)
        if not mel_filepath.is_file():
            mel_filepath = self.mel_dir / filepath

        if self._cache_ref_count == 0:
            audio = np.load(audio_filepath)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        if not self.fine_tuning:
            raise ValueError("Non-fine-tuning training not supported.")
        else:
            mel = np.load(mel_filepath).T

            # self.split = True : training
            # self.split = False : validation/inference (batch size = 1)
            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.shape[0] >= self.segment_size:
                    mel_start = random.randint(0, mel.shape[1] - frames_per_seg - 1)
                    mel = mel[:, mel_start:mel_start + frames_per_seg]
                    audio = audio[mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = np.pad(mel, ((0,0), (0, frames_per_seg - mel.shape[1])), mode='constant')
                    audio = np.pad(audio, (0, self.segment_size - audio.shape[0]), mode='constant')
            else:
                audio = np.pad(audio, (0, mel.shape[1] * self.hop_size - audio.shape[0]), mode='constant')

        audio = torch.FloatTensor(audio)
        mel_loss = mel_spectrogram(audio.unsqueeze(0), self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=self.win_center)
        mel = torch.FloatTensor(mel)

        return (mel.squeeze(), audio.squeeze(), mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
