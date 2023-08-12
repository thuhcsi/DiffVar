import json

import numpy as np
from librosa.core import load as loadwav
from librosa.core import stft
from librosa.filters import mel as mel_basis
from librosa.util import normalize
from scipy import signal


def _wav_addpadding(wav, hop_size:int, pad_mode:int, pad_val:float):
    pad_size = (wav.shape[0] // hop_size + 1) * hop_size - wav.shape[0]
    if pad_mode == 1:
        l_pad = 0
        r_pad = pad_size
    else:
        l_pad = pad_size // 2
        r_pad = pad_size // 2 + pad_size % 2
    return np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=pad_val)

def _wav_preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def _spec_normalize(S, clip:bool, symmetric:bool, max_val:float, min_db:float):
	if clip:
		if symmetric:
			return np.clip((2 * max_val) * ((S - min_db) / (-min_db)) - max_val,
			 -max_val, max_val)
		else:
			return np.clip(max_val * ((S - min_db) / (-min_db)), 0, max_val)

	assert S.max() <= 0 and S.min() - min_db >= 0

	if symmetric:
		return (2 * max_val) * ((S - min_db) / (-min_db)) - max_val
	else:
		return max_val * ((S - min_db) / (-min_db))

def wav2mel_npy(wav, sr=16000, \
    wav_pad=True, wav_pad_mode=1, wav_pad_val=0., \
    wav_rescale=False, wav_rescale_max=0.95, \
    pre_emph=True, pre_emph_cof=0.85, \
    n_fft=2048, hop_size=200, win_size=800, mag_pow=1., \
    n_mels=80, fmin=0., fmax=8000., \
    spec_ref_db=20, spec_min_db=-115, \
    spec_norm=True, spec_max=4., spec_sym=True, spec_clip=True):
    """ Extract Mel-spectrogram from wav file

    Parameters
    ----------
        wavpath : string, pathlib.Path object
            path of the source wavfile

        sr : int
            target sampling rate

        wav_pad : bool
            whether to pad the input waveform for STFT
        
        wav_pad_mode : integer with value 0 or 1
            0: add paddings to both sides of waveform
            1: add padding to the tail of waveform
        
        wav_pad_val : float
            value used as padding
            (default: zero-padding)

        wav_rescale : bool
            whether to normalize input waveform into
            range [0, @wav_rescale_max]

        wav_rescale_max : float
            upper bound of the rescaled waveform

        pre_emph : bool
            whether to pre-emphasize waveform before STFT
        
        pre_emph_cof : float
            coefficient of pre-emphasis
        
        n_fft : int > 0
            length of the windowed signal after padding with zeros

        hop_size : int > 0
            number of audio samples between adjacent STFT columns

        win_size : int > 0
            Each frame of audio is windowed by `window()` of length `win_length`
            and then padded with zeros to match `n_fft`
        
        mag_pow : float > 0
            Exponent for the magnitude spectrogram,
            e.g., 1 for energy, 2 for power, etc.

        n_mels : int > 0
            number of Mel filter bands

        fmin : float >= 0
            lowest frequency (in Hz) of Mel filter

        fmax : float >= 0
            highest frequency (in Hz) of Mel filter
            If `None`, use `fmax = sr / 2.0`

        spec_ref_db : int
            "reference" level of mel spectrogram in dB,
            which is subtracted from the raw spectrogram
        
        spec_min_db : int
            minimum of the spectrogram in dB
            values below it are clipped

        spec_norm : bool
            whether to normalize generated spectrogram

        spec_max : float
            the supremum of the absolute value of normalized spectrogram

        spec_sym : bool
            whether to normalize spectrogram to a range
            that is symmetric by 0.
            i.e. eihter [-spec_max, spec_max], or [0, spec_max]

        spec_clip : bool
            whethter to allow clipping while doing normalization

    Returns
    -------
        mel: np.ndarray [shape=(#spec_frame, @n_mels)]
            Extracted Mel spectrogram

        linear: np.ndarray [shape=(#spec_frame, #n_freq), with #n_freq = @n_fft / 2 + 1]
            Extracted Linear spectrogram

        wav: np.ndarray [shape=(#sample_point,)]
            Waveform after preprocessing
    
    """

    # 1. if a batch of wave array is provided
    #       do wav2mel for all wav samples in one single batch
    if len(wav.shape) > 1:
        result = [wav2mel_npy(single_wav) for single_wav in wav]
        return [np.stack(items) for items in list(zip(*result))]

    # 2. waveform preprocessing
    # 2.1 [optional] padding
    if wav_pad:
        wav = _wav_addpadding(wav, \
            hop_size=hop_size, pad_mode=wav_pad_mode, pad_val=wav_pad_val)
    
    # 2.2 [optional] normalization
    if wav_rescale:
        wav = normalize(wav) * wav_rescale_max

    # 3. spectrogram generation
    # 3.1 [optional] preemphasize waveform
    if pre_emph:
        pre_emphed_wav = _wav_preemphasis(wav, pre_emph_cof)
    else:
        pre_emphed_wav = wav

    # 3.2 magnitude spectrogram retrieve
    S = np.abs( \
            stft(pre_emphed_wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
        ).T ** mag_pow
    mel_S = np.dot(S, mel_basis(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T)

    # 3.3 convert magnitude spectrogram to dB-scaled units
    min_amp = np.power(10, spec_min_db/20)
    linear = 20 * np.log10(np.maximum(min_amp, S)) - spec_ref_db
    mel = 20 * np.log10(np.maximum(min_amp, mel_S)) - spec_ref_db

    # 3.4 [optional] spectrogram normalization
    if spec_norm:
        mel = _spec_normalize(mel, clip=spec_clip, symmetric=spec_sym, \
            max_val=spec_max, min_db=spec_min_db)
        linear = _spec_normalize(linear, clip=spec_clip, symmetric=spec_sym, \
            max_val=spec_max, min_db=spec_min_db)
    
    return mel, linear, wav

def wav2mel(wavpath, sr=16000, **hparams):
    # load wavfile into float waveform
    wav, _sr = loadwav(wavpath, sr=sr)
    return wav2mel_npy(wav, sr=_sr, **hparams)


def wav2mel_config(wavpath, config_path):
    with open(config_path, 'r') as f:
        hparams = json.load(f)
    return wav2mel(wavpath, **hparams)



