# mel_extractor
Extract Mel & Linear spectrogram from wav file. (Depending on `librosa` library)

There are 3 functions provided in this module to help extract **Mel-spectrogram** and **Linear spectrogram** from given input waveform.

1. `wav2mel_npy(wav, **hparams)`

    **Basic function to extract spectrogram from one, or a batch of wavs.**

    It **takes**:
    - input wav(s) in `numpy.ndarray` (typically loaded with `librosa`), whose shape might be:
        - `(#sample_points,)`: for one single wav
        - or `(#batch_size, #sample_points)`: for one batch of wavs
            - **DO note that**: input wavs within the same batch need to be reformed(clipping/padding) into the same shape, before stacked together.
            - Currently, wavs within one single batch are handle one by one. To process a large amount of wave files, the following function `wav2mel` and Python Concurrency features are recommended.
    - parameters related to the spectrogram extracting process (for details about each parameter item, please refer to the comment about the function in script `mel.py`)
    
    and **returns** a tuple made up by 3 `numpy.ndarray` objects:
    - the extracted Mel-spectrogram, whose shape is `([#batch_size, ]#spec_frame, @n_mels)`
    - the extracted Linear spectrogram, whose shape is `([#batch_size, ]#spec_frame, @n_freq)`
    - the waveform data (after preprocess), whose shape is `([#batch_size, ]#sample_points,)`
    

2. `wav2mel(wavpath, **hparams)`

    **A wrapped up version of the previous function. Usually used for dataset preprocess.**

    Before calling `wav2mel_npy`, the waveform data is first load from wave file with `librosa`.
    

3. `wav2mel_config(wavpath, config_path)`

    **A wrapped up version of the previous function.**
    
    Instead of taking dozens of parameters, it only requires the path of the **JSON** configuration file (`config16k.json` and `config22k.json` are shown as examples).

