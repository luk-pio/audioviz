import librosa
import numpy as np
from skimage.measure import block_reduce

from src.common.fun_call import FunCallFactory


def audioviz_stft(
    y,
    n_fft=1024,
    hop_length=None,
    use_logamp=False,  # boost the brightness of quiet sounds
    reduce_rows=10,  # how many frequency bands to average into one
    reduce_cols=1,  # how many time steps to average into one
    crop_rows=32,  # limit how many frequency bands to use
    crop_cols=32,  # limit how many time steps to use):
):
    # implementation from https://github.com/kylemcdonald/AudioNotebooks/blob/master/Samples%20to%20Fingerprints.ipynb
    window = np.hanning(n_fft)
    hop_length = n_fft // 4 if not hop_length else hop_length

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    amp = np.abs(S)
    if reduce_rows > 1 or reduce_cols > 1:
        amp = block_reduce(amp, (reduce_rows, reduce_cols), func=np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols - amp.shape[1])), "constant")
    amp = amp[:crop_rows, :crop_cols]
    if use_logamp:
        amp = librosa.logamplitude(amp ** 2)
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp)  # for visualization, put low frequencies on bottom

    return np.asarray(amp)


def audioviz_rms(y, n_fft=1024, hop_length=None):
    hop_length = n_fft // 4 if not hop_length else hop_length
    S = librosa.magphase(
        librosa.stft(
            y, n_fft=n_fft, hop_length=hop_length, window=np.ones, center=False
        )
    )[0]
    return librosa.feature.rms(S=S, frame_length=n_fft)


def audioviz_Spectral_Centroid(y, sr=22050, n_fft=1024, hop_length=None):
    hop_length = n_fft // 4 if not hop_length else hop_length
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )


def audioviz_Spectral_Crest(y, n_fft=1024, hop_length=None):
    hop_length = n_fft // 4 if not hop_length else hop_length
    sample_fft = librosa.core.stft(y, hop_length=hop_length)
    sample_fft = np.abs(sample_fft)
    numBlocks = sample_fft.shape[1]
    spectral_crest = np.zeros((1, numBlocks))
    for i in range(numBlocks):
        blockFFT = sample_fft[:, i]
        if np.sum(blockFFT) == 0:
            spectral_crest[
                :, i
            ] = 0  # Need to ask about this!!! Prevents division by zero
        else:
            spectral_crest[:, i] = np.max(blockFFT) / np.sum(blockFFT)
    return spectral_crest


def audioviz_Spectral_Flux(y, n_fft=1024, hop_length=None):
    hop_length = n_fft // 4 if not hop_length else hop_length
    sample_fft = librosa.core.stft(y, hop_length=hop_length)
    sample_fft = np.abs(sample_fft)
    (blockSize, numBlocks) = sample_fft.shape
    spectral_flux = np.zeros((1, numBlocks))
    spectral_flux[0] = 0  # Set flux at 0 for very first timestep
    for i in range(1, numBlocks):
        spectral_diff = sample_fft[:, i] - sample_fft[:, i - 1]
        spectral_flux[:, i] = np.sqrt(np.sum(np.square(spectral_diff))) / (blockSize)
    return spectral_flux


def audioviz_Spectral_Roll(y, sr=22050, n_fft=1024):
    return librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft)


def audioviz_Zerocrossing_Rate(y, frame_length=64):
    return librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length)


def audioviz_mfcc(
    y, st=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, hop_length=256,
):
    return librosa.feature.mfcc(
        y,
        st,
        S,
        n_mfcc=n_mfcc,
        dct_type=dct_type,
        norm=norm,
        lifter=lifter,
        hop_length=hop_length,
    )


class FeatureExtractorFactory(FunCallFactory):
    _implemented = {
        "stft": audioviz_stft,
        "mfcc": audioviz_mfcc,
        "rms": audioviz_rms,
        "Spectral_Centroid": audioviz_Spectral_Centroid,
        "Spectral_Crest": audioviz_Spectral_Crest,
        "Spectral_Flux": audioviz_Spectral_Flux,
        "Spectral_Roll": audioviz_Spectral_Roll,
        "Zerocrossing_Rate": audioviz_Zerocrossing_Rate,
    }
