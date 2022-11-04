import scipy
import copy
import numpy as np


def gesd(x, **kwargs):
    x_ = np.array(x)
    alpha = 0.05 if 'alpha' not in kwargs else kwargs['alpha']
    n_out = int(np.ceil(len(x_) * 0.1)) if 'n_out' not in kwargs else kwargs['n_out']
    outlier_side = 0 if 'outlier_side' not in kwargs else kwargs['outlier_side']
    alpha_ = alpha / 2 if outlier_side == 0 else alpha

    n = len(x_)
    temp = x_
    R = np.zeros([n_out])
    rm_idx = copy.copy(R).astype(int)
    lam = copy.copy(R)

    for j in range(n_out):
        if outlier_side == -1:
            sample = np.nanmin(temp)
            rm_idx[j] = list(temp).index(sample)
            R[j] = (np.nanmean(temp) - sample)
        elif outlier_side == 0:
            R[j] = np.nanmax(abs(temp - np.nanmean(temp)))
            rm_idx[j] = np.argmax(abs(temp - np.nanmean(temp)))
        else:
            sample = np.nanmax(temp)
            rm_idx[j] = list(temp).index(sample)
            R[j] = (sample - np.nanmean(temp))
        R[j] /= np.nanstd(temp)
        temp[rm_idx[j]] = float('nan')

        p = 1 - alpha_ / (n - j + 1)
        t = scipy.stats.t.ppf(p, n - j - 1)
        lam[j] = ((n - j) * t) / (np.sqrt((n - j - 1 + t ** 2) * (n - j + 1)))

    idx = np.zeros(n).astype(bool)
    if True in list(R > lam)[::-1]:
        a_ = list(R > lam)[::-1].index(True)
        b = rm_idx[0:a_]
        idx[b] = True
    x2 = x_[~idx]

    return idx, x2
