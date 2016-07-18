import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy import mean
import scikits.bootstrap as bootstrap


def moving_average(x, win=10):
    w = np.blackman(win)
    s = np.r_[2 * x[0] - x[win:1:-1], x, 2 * x[-1] - x[-1:-win:-1]]
    return np.convolve(w / w.sum(), s, mode='same')[win - 1: -win + 1]


def perpendicular(a):
    """
    gets perpendicular vector
    :rtype: array like
    :param a:
    :return:
    """
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:, 0]
    return b


def line(x, m, b):
    return m * x + b


def zero_crossings(data):
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def tribocycle_finder(y_disp):
    """
    finds tribo-cycles using the y displacement curve
    :param y_disp:
    :return:
    """

    y_disp = moving_average(y_disp)
    maxima = argrelextrema(y_disp, np.greater, order=1000)
    minima = argrelextrema(y_disp, np.less, order=500)

    if maxima[0].size > minima[0].size:
        cycle_ends = maxima
        cycle_mid = minima
    elif minima[0].size > maxima[0].size:
        cycle_ends = minima
        cycle_mid = maxima
    else:
        print 'Error in tribocycle finder, y displacement waveform incorrect'
        plt.plot(y_disp)
        plt.show()
        cycle_ends = np.nan
        cycle_mid = np.nan

    return cycle_ends, cycle_mid


def prony_min_hold(params, *args):
    """
    Description: Objective function for the 1-term prony analysis

    :param params:
        t = time vector
        F = axial force
        t1 = initial time
        e1 = initial displacement
        E = instantaneous modulus

    """
    t = args[0]
    F = args[1]
    t1 = args[2]
    e1 = args[3]
    E = args[4]

    p, tau = params
    F_model = (E * e1 / t1) * (t1 - p * t1 + p * tau * np.exp(-1 * (t - t1) / tau)
                               - p * tau * np.exp(-1 * t / tau))
    error = F - F_model
    return np.sum(error ** 2)


def prony_hold(t, E, p, tau, e1, t1):
    out = (E * e1 / t1) * (t1 - p * t1 + p * tau * np.exp(-1 * (t - t1) / tau) -
                           p * tau * np.exp(-1 * t / tau))

    return out


def loo_regression(x, y):
    """
    Tests for an outlier in a regression
    :param x: X values for regression
    :param y: Y values for regression
    :return: object numpy.ndarray -- reg_coeffs
    """
    # check that x and y are vectors and the same length
    if not x.shape[0] == y.shape[0]:
        print("x and y are not the same size")

    x = np.array(x)
    y = np.array(y)

    vlen = x.shape[0]
    reg_coeffs = np.zeros(vlen, dtype=float)
    mask = np.ones(vlen, dtype=bool)

    for i in xrange(vlen):
        mask[i] = False
        ss_x = x[mask]
        ss_y = y[mask]
        ss_mu = np.polyfit(ss_x, ss_y, deg=1)
        mask[i] = True
        reg_coeffs[i] = ss_mu[0]

    return reg_coeffs


def flag_outlier(in_vec, thresh_percentage=95):
    """
    Flags an outlier according to a percent difference threshold
    :param thresh_percentage: percent confidence interval
    :param in_vec:
    :return: outlier_ind
    """
    in_vec = np.array(in_vec)

    # find largest outlier
    outlier_ind = 0
    l2_resid_old = 0
    mask = np.ones(len(in_vec), dtype=bool)
    for i in xrange(in_vec.shape[0]):
        mask[i] = False
        l2_resid = (in_vec[i] - np.mean(in_vec[mask]))**2

        if l2_resid > l2_resid_old:
            outlier_ind = i

        l2_resid_old = l2_resid
        mask[i] = True

    # check if outlier is outside threshold percentage
    # bootstrap a 95% ci from data
    a_lvl = 1 - (thresh_percentage / 100.)
    CIs = bootstrap.ci(data=in_vec, statfunction=mean, alpha=a_lvl)
    if in_vec[outlier_ind] < CIs[0] or in_vec[outlier_ind] > CIs[1]:
        return outlier_ind
    else:
        return None


def remove_outlier_friction(x, y, thresh=95):
    mu_list = loo_regression(x, y)
    oli = flag_outlier(mu_list, thresh_percentage=thresh)

    if oli:
        mask = np.ones(len(mu_list), dtype=bool)
        mask[oli] = False
        mu = np.mean(mu_list[mask])

    else:
        mu = np.mean(mu_list)

    return mu, oli
