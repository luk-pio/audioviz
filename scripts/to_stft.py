n_fft = 1024
hop_length = n_fft/4
use_logamp = True # boost the brightness of quiet sounds
reduce_rows = 10 # how many frequency bands to average into one
reduce_cols = 1 # how many time steps to average into one
crop_rows = 32 # limit how many frequency bands to use
crop_cols = 32 # limit how many time steps to use
limit = None 

from os.path import join
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from multiprocessing import Pool
import numpy as np
import librosa

from utils import CLASSNAMES, SAMPLE_DIR, FEATURE_DIR

sample_fingerprints = {}
samples = {}
for d in CLASSNAMES:
    samples[d] = np.load(join(SAMPLE_DIR, d+'_samples.npy'))

window = np.hanning(n_fft)
def job(y):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    amp = np.abs(S)
    if reduce_rows > 1 or reduce_cols > 1:
        amp = block_reduce(amp, (reduce_rows, reduce_cols), func=np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols-amp.shape[1])), 'constant')
    amp = amp[:crop_rows, :crop_cols]
    if use_logamp:
        amp = librosa.amplitude_to_db(amp**2)
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp) # for visualization, put low frequencies on bottom
    return amp

for d in CLASSNAMES:
    pool = Pool()
    fingerprints = pool.map(job, samples[d][:limit])
    # fingerprints = job(samples[d][:limit])
    fingerprints = np.asarray(fingerprints).astype(np.float32)
    sample_fingerprints[d] = fingerprints
    print "generated finger print for", d, fingerprints.shape

for d in CLASSNAMES:
    np.save(join(FEATURE_DIR, d+'_stft.npy'), sample_fingerprints[d])
    print "saved", d+'_stft.npy'
