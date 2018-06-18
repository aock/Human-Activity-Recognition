import numpy as np
from src.dataManager import DataManager
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from src.filters import *



if __name__ == "__main__":

    # Filter requirements.
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    dm = DataManager(datafolder='data', test_size=0.0)

    X_train, X_test, Y_train, Y_test, class_counter = dm.load_random(num_data=1)

    # X_train = butter_lowpass_filter(X_train, cutoff, fs, order)

    # reshape data
    X = X_train[0,:,3]
    Y = X_train[0,:,4]
    Z = X_train[0,:,5]

    t = np.arange( 0,X.shape[0] )

    ### filter data
    ## butter lowpass

    # lX = butter_lowpass_filter(X, cutoff, fs, order)
    # lY = butter_lowpass_filter(Y, cutoff, fs, order)
    # lZ = butter_lowpass_filter(Z, cutoff, fs, order)
    lX = X_train[0,:,6]
    lY = X_train[0,:,7]
    lZ = X_train[0,:,8]
    ## fourier transform
    fX = np.fft.fft(lX)
    fY = np.fft.fft(lY)
    fZ = np.fft.fft(lZ)


    ## sample frequencies
    freqX = np.fft.fftfreq(len(lX), d= (t[1]-t[0]) * 20.0 )
    freqY = np.fft.fftfreq(len(lY), d= (t[1]-t[0]) * 20.0 )
    freqZ = np.fft.fftfreq(len(lZ), d= (t[1]-t[0]) * 20.0 )

    ## Plotting
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, X, 'rx', label='X')
    plt.plot(t, Y, 'gx', label='Y')
    plt.plot(t, Z, 'bx', label='Z')
    plt.plot(t, lX, 'r-', label='X lowpass')
    plt.plot(t, lY, 'g-', label='Y lowpass')
    plt.plot(t, lZ, 'b-', label='Z lowpass')
    plt.legend()

    plt.subplot(212)
    plt.plot(freqX, abs(fX)**2, 'r', label='X') ## will show a peak at a frequency of 1 as it should.
    plt.plot(freqY, abs(fY)**2, 'g', label='Y')
    plt.plot(freqZ, abs(fZ)**2, 'b', label='Z')
    plt.legend()
    plt.show()