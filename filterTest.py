import numpy as np
from src.dataManager import DataManager,DataGenerator
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from src.filters import *
from scipy.fftpack import fft

# butter config
order = 6
fs = 50.0       # sample rate, Hz
cutoff = 2.0  # desired cutoff frequency of the filter, Hz

def butter_low_filter(xs,ys,zs):
    xs_norm = normalize_mean(xs)
    ys_norm = normalize_mean(ys)
    zs_norm = normalize_mean(zs)
    # butter filter

    xs_l = butter_lowpass_filter(xs_norm, cutoff, fs, order)
    ys_l = butter_lowpass_filter(ys_norm, cutoff, fs, order)
    zs_l = butter_lowpass_filter(zs_norm, cutoff, fs, order)
    return xs_l,ys_l,zs_l

if __name__ == "__main__":

    # Filter requirements.


    # Get the filter coefficients so we can check its frequency response.

    dg = DataGenerator(datafolder='data/NRM', test_size=0.0, filter={})

    train_gen = dg.get_next_batch(batch_size=1)
    X_train,y = next(train_gen)

    # X_train = butter_lowpass_filter(X_train, cutoff, fs, order)

    # reshape data
    X = X_train[0,:,0]
    Y = X_train[0,:,1]
    Z = X_train[0,:,2]

    t = np.arange(0, X.shape[0])

    ### filter data
    ## butter lowpass

    # lX = butter_lowpass_filter(X, cutoff, fs, order)
    # lY = butter_lowpass_filter(Y, cutoff, fs, order)
    # lZ = butter_lowpass_filter(Z, cutoff, fs, order)

    ## fourier transform
    # fX = np.fft.fft(X)
    # fY = np.fft.fft(Y)
    # fZ = np.fft.fft(Z)
    # scipy
    # N = X.shape[0]
    # N*T = X.shape[0]
    # -> n = X.shape[0], T = 1
    T = 1
    N = X.shape[0]
    fT = np.linspace(0.0, 1/(2.0*T), N/2.0 )
    fX = fft(X)
    fY = fft(Y)
    fZ = fft(Z)

    X_freq_in_hertz = np.abs(fX[np.argmax(np.abs(fX))])
    Y_freq_in_hertz = np.abs(fY[np.argmax(np.abs(fY))])
    Z_freq_in_hertz = np.abs(fZ[np.argmax(np.abs(fZ))])
    print(X_freq_in_hertz)
    print(Y_freq_in_hertz)
    print(Z_freq_in_hertz)

    # highpass
    hX = X - X[0]
    hY = Y - Y[0]
    hZ = Z - Z[0]

    # print('removing low freq X')
    # hX = butter_highpass_filter(hX, 3.667, fs, order=order)

    # print('removing low freq Y')
    # hY = butter_highpass_filter(hY, 3.667, fs, order=order)

    # print('removing low freq Z')
    # hZ = butter_highpass_filter(hZ, 3.667, fs, order=order)

    # lowpass
    lX = hX
    lY = hY
    lZ = hZ

    # removing higher than
    low_pass_freq = 2.0

    # removing lower than
    high_pass_freq = 2.0

    print('removing high freq X')
    lX = butter_lowpass_filter(lX, low_pass_freq, fs, order=order)

    print('removing high freq Y')
    lY = butter_lowpass_filter(lY, low_pass_freq, fs, order=order)

    print('removing high freq Z')
    lZ = butter_lowpass_filter(lZ, low_pass_freq, fs, order=order)



    # ## sample frequencies
    # freqX = np.fft.fftfreq(len(fX), d= 1.0/fs )
    # freqY = np.fft.fftfreq(len(fY), d= 1.0/fs )
    # freqZ = np.fft.fftfreq(len(fZ), d= 1.0/fs )

    # X_freq_in_hertz = np.abs(freqX[np.argmax(np.abs(fX))])
    # Y_freq_in_hertz = np.abs(freqY[np.argmax(np.abs(fY))])
    # Z_freq_in_hertz = np.abs(freqZ[np.argmax(np.abs(fZ))])
    # print('Main freq X: ' + str(X_freq_in_hertz) )
    # print('Main freq Y: ' + str(Y_freq_in_hertz) )
    # print('Main freq Z: ' + str(Z_freq_in_hertz) )

    # # butter low pass
    # xs_norm = normalize_mean(X)
    # ys_norm = normalize_mean(Y)
    # zs_norm = normalize_mean(Z)

    # lX = butter_lowpass_filter(xs_norm, X_freq_in_hertz, fs, order)
    # lY = butter_lowpass_filter(ys_norm, Y_freq_in_hertz, fs, order)
    # lZ = butter_lowpass_filter(zs_norm, Z_freq_in_hertz, fs, order)


    ## Plotting
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, X, 'rx', label='X')
    plt.plot(t, Y, 'gx', label='Y')
    plt.plot(t, Z, 'bx', label='Z')
    plt.plot(t, lX, 'r-', label='X lowpass')
    plt.plot(t, lY, 'g-', label='Y lowpass')
    plt.plot(t, lZ, 'b-', label='Z lowpass')
    # plt.plot(t, hX, 'r-', label='X highpass')
    # plt.plot(t, hY, 'g-', label='Y highpass')
    # plt.plot(t, hZ, 'b-', label='Z highpass')
    plt.legend()

    plt.subplot(212)
    plt.plot(fT, 2.0/N * np.abs(fX[:N//2]), 'r', label='X')
    plt.plot(fT, 2.0/N * np.abs(fY[:N//2]), 'g', label='Y')
    plt.plot(fT, 2.0/N * np.abs(fZ[:N//2]), 'b', label='Z')

    # plt.plot(freqX, abs(fX)**2, 'r', label='X') ## will show a peak at a frequency of 1 as it should.
    # plt.plot(freqY, abs(fY)**2, 'g', label='Y')
    # plt.plot(freqZ, abs(fZ)**2, 'b', label='Z')
    plt.legend()
    plt.show()