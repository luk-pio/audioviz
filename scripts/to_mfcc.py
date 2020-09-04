import numpy as np
import librosa
from tqdm import tqdm
import os

from utils import FEATURE_DIR, CLASSNAMES, SAMPLE_DIR

data_root = SAMPLE_DIR

samples = {}
for d in CLASSNAMES:
    samples[d] = np.load(os.path.join(data_root, d+'_samples.npy'))

# Sanity test:
pia = samples["pia"]
sax = samples["sax"]
# plt.plot(sax[10])

test_sax = sax[10]
test_sax_mfcc = librosa.feature.mfcc(test_sax,n_mfcc=30,sr=48000)
print(test_sax_mfcc.shape)

for classname in tqdm(CLASSNAMES):
    class_samples = samples[classname] 
    samples_mfcc = []
    (num_samples, sample_length) = class_samples.shape # e.g samples.shape=(672,12000)
    for i in tqdm(range(num_samples)):
        sample = class_samples[i]
        sample_mfcc = librosa.feature.mfcc(sample,n_mfcc=30,sr=48000)
        samples_mfcc.append(sample_mfcc)
    samples_mfcc = np.asarray(samples_mfcc)
    print (classname, samples_mfcc.shape)  
    file_path = os.path.join(FEATURE_DIR, classname + '_mfcc.npy')
    np.save(file_path, samples_mfcc)
