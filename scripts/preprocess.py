import os
from collections import defaultdict
from multiprocessing import Pool
from os.path import join

import librosa
import numpy as np


def find_files(root, file_types=None):
    if file_types is None:
        file_types = ['']
    # Collect all of the sample file paths as strings
    instrument_sets = defaultdict(list)
    for directory in os.walk(root):
        dirpath, dirnames, filenames = directory
        # get the dirname from path
        dirname = os.path.split(dirpath)[1]
        for filename in filenames:
            relative_path = dirpath + "/" + filename
            if any([filename.endswith(file_type) for file_type in file_types]):
                instrument_sets[dirname].append(relative_path)
    return instrument_sets


def load_sample(fn, sr=None, duration=None, normalize=True):
    if fn == '':  # ignore empty filenames
        return None
    audio, sr = librosa.load(fn, sr, mono=True, duration=duration)
    file_length = len(audio)
    if file_length == 0:  # ignore zero-length samples
        return None
    # in case files is shorter than duration, pad with zeros
    if file_length / sr < duration:
        audio.resize(duration * sr)
    max_val = np.abs(audio).max()
    if max_val == 0:  # ignore completely silent sounds
        return None
    if normalize:
        audio /= max_val
    return fn, audio, duration


def load_samples_multithread(instrument_sets, sr=None, duration=None, normalize=True, limit=None):
    def job(fn):
        return load_sample(fn, sr=sr, duration=duration, normalize=normalize)

    loaded_samples = {}
    for instrument, files in instrument_sets.items():
        pool = Pool()
        loaded_samples[instrument] = pool.map(job, files[:limit])
        print(('Processed', len(loaded_samples[instrument]), 'samples for ', instrument))
    return loaded_samples


def write_output(classnames, loaded_samples):
    for instrument in classnames:
        valid = [_f for _f in loaded_samples[instrument] if _f]
        filenames = [x[0] for x in valid]
        samples = [x[1] for x in valid]
        durations = [x[2] for x in valid]
        samples = np.asarray(samples)
        lengths.append(len(samples))
        with open(join(data_root, instrument + '_filenames.txt'), 'w+') as f:
            np.savetxt(f, filenames, fmt='%s')
        with open(join(data_root, instrument + '_durations.txt'), 'w+') as f:
            np.savetxt(f, durations, fmt='%i')
        with open(join(data_root, instrument + '_samples.npy'), 'w+') as f:
            np.save(f, samples)
        print('Saved', len(valid), 'samples of ' + instrument)


def main():
    lengths = []


if __name__ == '__main__':
    main()
# pickle.dump(CLASSNAMES, open(data_root+"/CLASSNAMES.pickle", "w"))
# pickle.dump(lengths, open(data_root+"/lengths.pickle", "w"))

# Regex matching for extracting secondary and tertiary attributes from filenames

# From the drum class names, generate the regular expression used to match against sample file paths
# regex = r"\d{3}__\[(\w{3})\]\[(\w{3})\]\[(\w+)\]\d+__\d+.wav"

# filter filenames into sets by matching vs regex
# instrument_sets = {}
# for i in range(len(CLASSNAMES)):
#     instrument_sets[CLASSNAMES[i]] = {fileName for fileName in filenames if re.match(drumRegex[i], fileName)}
