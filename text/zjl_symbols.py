_pad = "_"

_final = [f"{s}{i}" for s in ['u', 'o', 'iy', 'er', 'i', 'ng', 'v', 'e', 'a', 'n', 'ix'] for i in range(1,7)]
_init_mid = ['a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ng', 'o', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'v', 'x', 'z', 'zh']

_prosodic = [f'#{i}' for i in range(1, 5)]
_sil = [f'{sym}@{lv}' for sym in ['sil', 'pau'] for lv in list(range(5))+['S']]

_punc = ['!', '"', "'", '(', ')', ',', '.', ':', ';', '?', '—', '…', '、', '《', '》']

quasi_symbols = _punc + _prosodic

symbols = (
    [_pad]
    + _init_mid
    + _final
    + _prosodic
    + _sil
    + _punc
)