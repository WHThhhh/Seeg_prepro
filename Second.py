import scipy
import numpy as np

path = 'E:/BaiduNetdiskDownload/sub-songxingjiu/Result/'
Ripple_ind = np.load(path + 'D2_Ripple_peak_ind.npy', allow_pickle=True)
Ripple_epochs = np.load(path + 'D2_Ripple_epochs.npy', allow_pickle=True)
Spindle_ind = np.load(path + 'D2_Spindle_peak_ind.npy', allow_pickle=True)
Spindle_epochs = np.load(path + 'D2_Spindle_epochs.npy', allow_pickle=True)
SO_ind = np.load(path + 'D2_SO_down_ind.npy', allow_pickle=True)
SO_epochs = np.load(path + 'D2_SO_epochs.npy', allow_pickle=True)
Raw_data = np.load(path + 'D2_raw_data.npy', allow_pickle=True)

eeg_chan = list(range(10, 16))

Couple_l = []
Couple_epochs = []
EEG_couple_l = []
for i in range(len(Ripple_ind)):
    R, S, O = Ripple_ind[i], Spindle_ind[i], SO_ind[i]
    Couple_chan = []
    Couple_ep_chan = []
    EEG_couple_chan = []
    for ii, r in enumerate(R):
        closest_S = min(S, key=lambda x: abs(x - r))
        closest_O = min(O, key=lambda x: abs(x - r))
        if abs(closest_S - r) <= 200 and abs(closest_O - r) <= 200 and abs(closest_O - closest_S) <= 200:
            s = S.index(closest_S)
            o = O.index(closest_O)
            Couple_chan.append([closest_O, closest_S, r])
            Couple_ep_chan.append(Ripple_epochs[i][ii])
            EEG_couple_chan.append(Raw_data[10:16, r-750:r+750])
    EEG_couple_l.append(EEG_couple_chan)
    Couple_l.append(Couple_chan)
    Couple_epochs.append(Couple_ep_chan)

for i in eeg_chan:
    S, O = Spindle_ind[i], SO_ind[i]
    Couple_chan = []
    Couple_ep_chan = []
    for ii, s in enumerate(S):
        closest_O = min(O, key=lambda x: abs(x - s))
        if abs(closest_O - s) <= 200:
            o = O.index(closest_O)
            Couple_chan.append([closest_O, s, 0])
            Couple_ep_chan.append(Spindle_epochs[i][ii])

    Couple_l.append(Couple_chan)
    Couple_epochs.append(Couple_ep_chan)

scipy.io.savemat('./Result/Three_Couple_epochs.mat', {'Couple_epochs': Couple_epochs, 'EEG_couple': EEG_couple_l,
                                                      'Couple_ind': Couple_l})
