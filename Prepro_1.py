import glob
import torch
import scipy
import torch.nn.functional as F
import numpy as np


def bandpass_wht(sig, lf, hf, fs):
    b, a = scipy.signal.butter(3, [2 * lf / fs, 2 * hf / fs], 'bandpass')
    sig_n = scipy.signal.filtfilt(b, a, sig)
    return sig_n


def moving_window_rms_wht(sig, window):
    sig = torch.tensor(np.power(sig, 2), dtype=torch.float32).unsqueeze(1).to('cpu')
    weight = (torch.ones([1, 1, window]) / torch.tensor(window, dtype=torch.float32)).to('cpu')
    return F.conv1d(sig, weight).to('cpu')


def find_len(sig, min_len, max_len, raw_data, rms_win):
    sig_diff = sig[:, 1:] - sig[:, :-1]
    peak_l = []
    for i in range(sig_diff.shape[0]):
        peak_chan = []
        diff_i = torch.where(sig_diff[i, :] != 0)[0]
        last_p = 0
        for p in diff_i:
            if sig_diff[i, p] == 1:
                last_p = p - rms_win // 2
            elif sig_diff[i, p] == -1:
                p_ = p - rms_win // 2
                if min_len <= (p_ - last_p) <= max_len:
                    peak_ind = list(abs(raw_data[i, last_p:p_ + 1])).index(
                        max(abs(raw_data[i, last_p:p_ + 1]))) + last_p
                    peak_chan.append(int(peak_ind))
        peak_l.append(peak_chan)

    return peak_l


def find_SO(sig, is_eeg):
    SO_1 = []
    for i, s in enumerate(sig):
        SO_chan = []
        Am_chan = []
        n2p = p2n = 0
        for ii, p in enumerate(s[:-1]):
            if p * s[ii + 1] < 0 and p < 0:
                if not is_eeg and n2p != 0 and 400 <= (ii - n2p) <= 1000:
                    temp = sig[i, n2p:ii + 1]
                    m = abs(max(temp) - min(temp))
                    down_ind = list(temp).index(min(temp)) + n2p
                    SO_chan.append(down_ind)
                    Am_chan.append(m)
                n2p = ii
            elif p * s[ii + 1] < 0 and p > 0:
                if is_eeg and p2n != 0 and 400 <= (ii - p2n) <= 1000:
                    temp = sig[i, p2n:ii + 1]
                    m = abs(max(temp) - min(temp))
                    down_ind = list(temp).index(max(temp)) + p2n
                    SO_chan.append(down_ind)
                    Am_chan.append(m)
                p2n = ii
        Am_threshold = np.percentile(Am_chan, 80)
        SO_chan_1 = [SO_chan[iii] for iii, x in enumerate(Am_chan) if x >= Am_threshold]
        SO_1.append(SO_chan_1)

    return SO_1


def SO_epochs_save(sig, is_eeg, name):
    SO_ = bandpass_wht(sig, 0.16, 1.25, Fs)
    SO_ind = find_SO(SO_, is_eeg)
    SO_epochs = Epochs_wht(SO_ind, sig, 750)
    np.savez('./Result/' + file_name[-2:] + '_' + name + '_SO.npz', SO_ind=SO_ind, SO_epochs=SO_epochs)


def Spindle_epochs_save(sig, name):
    Spindle_ = bandpass_wht(sig, 12, 16, Fs)
    Spindle_rms = moving_window_rms_wht(Spindle_, 100)
    Spindle_threshold = torch.quantile(Spindle_rms, 0.80, dim=2).unsqueeze(1)
    Spindle_label_ori = (Spindle_rms >= Spindle_threshold).float().squeeze()
    Spindle_peak = find_len(Spindle_label_ori, 250, 1500, Spindle_, 100)
    Spindle_epochs = Epochs_wht(Spindle_peak, sig, 750)
    np.savez('./Result/' + file_name[-2:] + '_' + name + '_Spindle.npz', Spindle_ind=Spindle_peak,
             Spindle_epochs=Spindle_epochs)


def Ripple_epochs_save(sig, name):
    Ripple_ = bandpass_wht(sig, 80, 100, Fs)
    Ripple_rms = moving_window_rms_wht(Ripple_, 10)
    Ripple_threshold = torch.quantile(Ripple_rms, 0.99, dim=2).unsqueeze(1)
    Ripple_label_ori = (Ripple_rms >= Ripple_threshold).float().squeeze()
    Ripple_peak = find_len(Ripple_label_ori, 19, 1e10, Ripple_, 10)
    Ripple_epochs = Epochs_wht(Ripple_peak, sig, 750)
    np.savez('./Result/' + file_name[-2:] + '_' + name + '_Ripple.npz', Ripple_ind=Ripple_peak,
             Ripple_epochs=Ripple_epochs)


def Epochs_wht(sig_peak_ind, raw_data, time_range):
    All_epochs = []
    for i, sig in enumerate(sig_peak_ind):
        chan_epochs = []
        for p in sig:
            chan_epochs.append(raw_data[i, p - time_range:p + time_range])
        All_epochs.append(chan_epochs)
    return All_epochs


path = './Result/'
# /lustre/grp/gjhlab/lvbj/lyz_grp/wht/Three_coupling_prepro/
# E:/BaiduNetdiskDownload/sub-songxingjiu/
Files = glob.glob(path + '/*.npz')

for file in Files:
    file_name = file.split("_")[-3]
    Data = np.load(file)
    hippo_data = Data['hippo_data']
    eeg_data = Data['eeg_data']
    Fs = 512
    del Data

    SO_epochs_save(hippo_data, False, 'hippo')
    SO_epochs_save(eeg_data, True, 'eeg')
    Spindle_epochs_save(hippo_data, 'hippo')
    Spindle_epochs_save(eeg_data, 'eeg')
    Ripple_epochs_save(hippo_data, 'hippo')
