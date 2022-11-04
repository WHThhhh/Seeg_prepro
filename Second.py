import scipy
import numpy as np
import glob


def load_each_files(chan_type, sub_):
    ripple_data = [file for file in sub_ if chan_type + '_Ripple' in file]
    spindle_data = [file for file in sub_ if chan_type + '_Spindle' in file]
    SO_data = [file for file in sub_ if chan_type + '_SO' in file]
    SO = np.load(SO_data[0], allow_pickle=True)
    Spindle = np.load(spindle_data[0], allow_pickle=True)
    Ripple = np.load(ripple_data[0], allow_pickle=True) if chan_type == 'hippo' else []
    return [SO, Spindle, Ripple]


path = './Result/'
All_files = glob.glob(path + '/*.npz')
sub_file_list = [[file for file in All_files if 'D2' in file], [file for file in All_files if 'D3' in file]]

for sub in sub_file_list:
    sub_name = sub[0].split('_')[-3][-2:]
    Raw_data = np.load([file for file in sub if 'data' in file][0])
    eeg_data = Raw_data['eeg_data'].astype(np.float32)
    hippo_SO, hippo_Spindle, hippo_Ripple = load_each_files('hippo', sub)
    hippo_Ripple_ind = hippo_Ripple['Ripple_ind']
    hippo_Spindle_ind = hippo_Spindle['Spindle_ind']
    hippo_SO_ind = hippo_SO['SO_ind']
    hippo_Ripple_epochs = hippo_Ripple['Ripple_epochs']

    eeg_SO, eeg_Spindle, _ = load_each_files('eeg', sub)
    eeg_SO_ind = eeg_SO['SO_ind']
    eeg_Spindle_ind = eeg_Spindle['Spindle_ind']
    eeg_Spindle_epochs = eeg_Spindle['Spindle_epochs']

    del hippo_SO, hippo_Spindle, hippo_Ripple, eeg_Spindle, eeg_SO, Raw_data

    Couple_l = []
    Couple_epochs = []
    EEG_couple_l = []
    for i in range(len(hippo_Ripple_ind)):
        R, S, O = hippo_Ripple_ind[i], hippo_Spindle_ind[i], hippo_SO_ind[i]
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
                Couple_ep_chan.append(hippo_Ripple_epochs[i][ii])
                EEG_couple_chan.append(eeg_data[:, r - 750:r + 750])
        EEG_couple_l.append(EEG_couple_chan)
        Couple_l.append(Couple_chan)
        Couple_epochs.append(Couple_ep_chan)

    for i, S in enumerate(eeg_Spindle_ind):
        O = eeg_SO_ind[i]
        Couple_chan = []
        Couple_ep_chan = []
        for ii, s in enumerate(S):
            closest_O = min(O, key=lambda x: abs(x - s))
            if abs(closest_O - s) <= 200:
                o = O.index(closest_O)
                Couple_chan.append([closest_O, s, 0])
                Couple_ep_chan.append(eeg_Spindle_epochs[i][ii])

        Couple_l.append(Couple_chan)
        Couple_epochs.append(Couple_ep_chan)

    scipy.io.savemat('./Result/' + sub_name + 'Three_Couple_epochs.mat', {'Couple_epochs': Couple_epochs,
                                                                          'EEG_couple': EEG_couple_l,
                                                                          'Couple_ind': Couple_l})
