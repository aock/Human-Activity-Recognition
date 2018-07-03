import numpy as np
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    """butterworth lowpass filter parameter extraction

    Arguments:
        cutoff {float} -- cutoff frequency (can be determined with fourier)
        fs {float} -- frequency of data

    Keyword Arguments:
        order {int} -- order of butterworth filter (default: {5})

    Returns:
        float -- parameter b
        float -- parameter a
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """butterworth lowpass filter

    Arguments:
        data {numpy.array} -- input data
        cutoff {float} -- frequency to cutoff (can be determined with fourier)
        fs {float} -- frequency of data

    Keyword Arguments:
        order {int} -- order of butterworth lowpass filter (default: {5})

    Returns:
        numpy.array -- filtered data
    """

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    """butterworth highpass filter parameter extraction

    Arguments:
        cutoff {float} -- cutoff frequency (can be determined with fourier)
        fs {float} -- frequency of data

    Keyword Arguments:
        order {int} -- order of butterworth filter (default: {5})

    Returns:
        float -- parameter b
        float -- parameter a
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """butterworth highpass filter

    Arguments:
        data {numpy.array} -- input data
        cutoff {float} -- frequency to cutoff (can be determined with fourier)
        fs {float} -- frequency of data

    Keyword Arguments:
        order {int} -- order of butterworth highpass filter (default: {5})

    Returns:
        numpy.array -- filtered data
    """

    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_first_point(X):
    """ Normalizing array with first point

    Arguments:
        X {numpy.array} -- input data

    Returns:
        numpy.array -- normalized output data
    """

    return X - X[0]

def normalize_mean(X):
    """ Normalizing array with mean

    Arguments:
        X {numpy.array} -- input data

    Returns:
        numpy.array -- normalized output data
    """

    return X - np.mean(X)