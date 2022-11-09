import scipy
import glob
import mne
import numpy as np
import pandas as pd
from GESD_wht import gesd


def hamming_fir_filter(sig, f_order, cutoff_, fs, mode='bandstop'):
    window = scipy.signal.firwin(f_order, cutoff=cutoff_, window='hamming', pass_zero=mode, fs=fs)
    filtered_sig = scipy.signal.filtfilt(window, 1, sig)
    return filtered_sig


def is_seeg_chan(chan_):
    Capital_alpha = list(range(65, 91))
    chan_label = np.zeros(len(chan_))
    for i, x in enumerate(chan_):
        chan_label[i] = 1 if (ord(x[0]) in Capital_alpha and x[1:].isdigit() and len(x) == 3) else 0
    seeg_chan = np.where(chan_label == 1)[0]
    eeg_chan = np.where(chan_label == 0)[0]
    return seeg_chan, eeg_chan


def detect_bad_channel(sig, max_bad_channels=10, max_iters=5):
    chan_ind = list(range(sig.shape[0]))
    detect_bad = True
    iters = 0
    bad_channels_ind = []
    good_chan_ind = chan_ind
    while detect_bad and iters < max_iters:
        iters += 1
        good_chan_dat = sig[good_chan_ind, :]
        if len(good_chan_ind) > 1 and len(bad_channels_ind) < max_bad_channels:
            if len(good_chan_dat.shape) == 3:
                dat = np.reshape(good_chan_dat,
                                 [good_chan_dat.shape[0], good_chan_dat.shape[1] * good_chan_dat.shape[2]])
            else:
                dat = good_chan_dat
            std_chan = np.std(dat, axis=1)
            idx, x2 = gesd(std_chan, alpha=0.05, n_out=max_bad_channels - len(bad_channels_ind), outlier_side=0)
            to_add = [n for n, a in enumerate(idx) if a]
            for c in to_add:
                bad_channels_ind.append(good_chan_ind[c])
            detect_bad = True if to_add else False
        else:
            detect_bad = False
        good_chan_ind = [ind for ind in chan_ind if ind not in bad_channels_ind]
    return bad_channels_ind, good_chan_ind


def z_score_remove(sig):
    sig_mean = np.mean(sig, axis=-1)
    sig_std = np.std(sig, axis=-1)
    out_min = np.expand_dims(np.array(sig_mean - 5 * sig_std), axis=1)
    out_max = np.expand_dims(np.array(sig_mean + 5 * sig_std), axis=1)
    sig_times = set(range(sig.shape[1]))
    rm_range = []
    rm_points = list(set(np.where(np.array((sig <= out_min).tolist() or (sig >= out_max).tolist()) == 1)[1]))
    rm_range += [list(range(x - 500, x + 500)) for x in rm_points]
    rm_range_1d = []
    for e in rm_range:
        rm_range_1d.extend(e)
    return sorted(list(sig_times.difference(set(rm_range_1d))))


def artifact_remove(hi, ee):
    hippo_remain = z_score_remove(hi)
    eeg_remain = z_score_remove(ee)
    return hippo_remain, eeg_remain


path = './data/'
# /lustre/grp/gjhlab/lvbj/lyz_grp/wht/Three_coupling_prepro/
# E:/BaiduNetdiskDownload/sub-songxingjiu/
Files = glob.glob(path + '/*.set')
chan_loc_files = glob.glob(path + '/*.tsv')
cutoff = [49, 51, 99, 101, 149, 151, 199, 201, 249, 251]
forder = 1689
for file in Files:
    file_name = file.split("-")
    sub_name = file_name[-3].split("_")[0]
    Data = mne.io.read_raw_eeglab(file, preload=True)
    data = Data._data
    chan = Data.ch_names
    Fs = 512
    seeg_chan_ind, eeg_chan_ind = is_seeg_chan(chan)
    seeg_data = data[seeg_chan_ind, :]
    eeg_data = data[eeg_chan_ind, :]

    del Data, data

    # eeg_filter
    eeg_data = hamming_fir_filter(eeg_data, forder, [0.1, 30], Fs, 'bandpass')
    eeg_data = scipy.signal.detrend(eeg_data, axis=-1)

    # remove seeg bad channels
    bad_chan_ind, good_chan_ind = detect_bad_channel(seeg_data - np.mean(seeg_data, axis=0))
    if bad_chan_ind:
        seeg_data_good = seeg_data[good_chan_ind, :]
        chan_good = chan[good_chan_ind]
    else:
        seeg_data_good = seeg_data
        chan_good = chan

    # hippocampus_seeg_rereference
    for x in chan_loc_files:
        if sub_name in x:
            chan_locs = pd.read_csv(x, sep='\t')
    ROI = list(chan_locs['ASEG'])
    MNI = list(chan_locs['MNI'])
    for i, s in enumerate(MNI):
        x = s.strip('[]').split(',')
        x = [float(ii) for ii in x]
        MNI[i] = x
    ROI_good = [R for n, R in enumerate(ROI) if n in good_chan_ind]
    MNI_good = [R for n, R in enumerate(MNI) if n in good_chan_ind]

    hippocampus_ind = [i for i, r in enumerate(ROI_good) if 'Hippocampus' in str(r)]

    hippocampus_site_names = [chan_good[i] for i in hippocampus_ind]
    hippocampus_chan_names = []
    for c in hippocampus_site_names:
        hippocampus_chan_names += c[0] if c[0] not in hippocampus_chan_names else []

    white_ind = [i for i, r in enumerate(ROI_good) if 'White' in str(r) and chan_good[i][0] in hippocampus_chan_names]
    white_locs = np.array([np.array(MNI_good[x]) for x in white_ind])

    ref_white = []
    for h in hippocampus_ind:
        loc_h = np.array(MNI_good[h])
        h_chan_name = chan_good[h][0]
        distant = list(np.sum(np.power(white_locs - loc_h, 2), axis=1))
        closet_white_ind = np.argmin(distant)
        while chan_good[white_ind[closet_white_ind]][0] != h_chan_name:
            distant[closet_white_ind] = float('inf')
            closet_white_ind = np.argmin(distant)
            if distant[closet_white_ind] == float('inf'):
                raise Exception("No white matter sites on this hippocampus channel!")
        ref_white.append(white_ind[closet_white_ind])

    hippo_data = seeg_data_good[hippocampus_ind, :] - seeg_data_good[ref_white, :]

    # rm line power noise
    hippo_data = hamming_fir_filter(hippo_data, forder, cutoff, Fs, 'bandstop')

    # Artifact detection
    # hippo_data, eeg_data = artifact_remove(hippo_data, eeg_data)
    # hippo_diff = hippo_data[:, :-1] - hippo_data[:, 1:]
    # eeg_diff = eeg_data[:, :-1] - eeg_data[:, 1:]
    # hippo_data, eeg_data = artifact_remove(hippo_diff, eeg_diff)

    np.savez('./Result/' + file_name[-2] + '_data_preprocessed.npz', hippo_data=hippo_data, eeg_data=eeg_data)
