# -*- coding: utf-8 -*-
# Imports
import scipy.io
import soundfile as sf
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Import helper functions
from model.utils.helper_functions import awgn, sliding_window, sliding_window_rec, Yule_Walker, kalman_ite


def load_kalmen(fileName):

    noisy_test, sr = sf.read(fileName)

    # scipy.io.savemat('./datasets/TIMIT/test/noisy/train/filename.mat', mdict={'noisy': noisy_test})
    # if (signal_number == 1):
    #     signal = scipy.io.loadmat('./datasets/TIMIT/test/noisy/fcno01fz.mat')  # 加载文件
    #     signal = np.array(signal["fcno01fz"])
    # else:
    #     signal = scipy.io.loadmat('fcno02fz.mat')
    #     signal = np.array(signal["fcno02fz"])
    # signal = scipy.io.loadmat('./datasets/TIMIT/test/noisy/train/filename.mat')
    # signal = np.array(signal["noisy"].T)
    signal = np.reshape(noisy_test, (noisy_test.shape[0], 1))
    Fs = 8000  # 8 kHz sample frequency

    # Signal length 初始化信号长度
    N = signal.shape[0]

    # time vector 时间向量
    t = np.arange(0, N / Fs, 1 / Fs)
    t = np.reshape(t, (len(t), 1))

    # frequency vector 频率向量
    freq = np.linspace(-Fs / 2, Fs / 2, N)
    freq = np.reshape(freq, (len(freq), 1))

    # Yules-Walker and AR parameters 参数设置
    p = 16  # AR order
    ite_kalman = 10  # We apply YW-KALMAN a few times to each slice

    # Noise addition 噪音加成
    SNR = 0  # dB
    noisy_signal, wg_noise = awgn(signal, SNR)

    # Plot noisy signal and original signal  标定噪声信号和原始信号
    # plt.figure()
    # plt.grid()
    # plt.title('Original signal')
    # plt.plot(t, signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend(['Original signal', 'Noisy signal'])
    # plt.show()
    #
    # plt.figure()
    # plt.grid()
    # plt.title('Original signal vs Noisy signal')
    # plt.plot(t, noisy_signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    # Noise variance estimation (using silence ) 噪声方差估计（用静音）
    varNoise = np.var(noisy_signal[1:4500])

    # Sliced signal 切割信号
    signal_sliced_windowed, padding = sliding_window(signal, Fs)

    # Save after filtering 滤波后保存
    signal_sliced_windowed_filtered = np.zeros((signal_sliced_windowed.shape))

    for ite_slice in range(signal_sliced_windowed.shape[1]):

        # Slice n
        slice_signal = signal_sliced_windowed[:, ite_slice:ite_slice + 1].T

        # On fait YW-KALMAN plusieurs tours pour chaque morceau
        for ite in range(ite_kalman):
            # YW
            a, var_bruit = Yule_Walker(slice_signal, p)
            #        ar, variance, coeff_reflection = aryule(slice_signal, p)

            # Save
            signal_filtered = np.zeros((1, signal_sliced_windowed.shape[0]))

            # Phi et H
            Phi = np.concatenate((np.zeros((p - 1, 1)), np.eye(p - 1)), axis=1)
            Phi = np.concatenate((Phi, -np.fliplr(a[1:].T)), axis=0)

            H = np.concatenate((np.zeros((p - 1, 1)), np.ones((1, 1))), axis=0).T

            # Q, R and Po
            Q = var_bruit * np.eye(p)
            R = varNoise
            P = 10000 * np.eye(p)

            # Initialisation vecteur d'etat
            x = np.zeros((p, 1))

            for jj in range(signal_sliced_windowed.shape[0]):
                y = slice_signal[0][jj]  # Observation
                [x, P] = kalman_ite(x, P, y, Q, R, Phi, H)
                signal_filtered[0][jj] = x[-1]

            slice_signal = signal_filtered
        signal_sliced_windowed_filtered[:, ite_slice:ite_slice + 1] = signal_filtered.T

    # Reconstruct signal
    signal_reconstructed = sliding_window_rec(signal_sliced_windowed_filtered, Fs, padding)

    # Plot reconstructed signal and original signal
    x = signal_reconstructed.reshape(-1)
    sf.write('/home/aone/PycharmProjects/kj-flask-system/static/enhanced/' + "kalmen" + fileName.split('/')[-1], x, sr)
