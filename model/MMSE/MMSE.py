import os
import math
import wave
import argparse
import scipy.io.wavfile as wav
import soundfile as sf
from model.utils.estnoise_ms import *
from model.utils.utils import *


# 全称Minimum Mean Square Error（最小均方误差）=>找出接收到的数据与原始数据尽可能接近的数据
# 采用MMSE-STSA(短时谱幅)=>非线性的
# 该算法假设观测到的语音信号和噪声不相关，利用计算所得到的噪声功率谱信息，从含有噪声的语音信号的频谱分量中估计出纯净语音的频谱分量，借助输入语音信号的相位得到增强后的语音信号


def load_MMSE(fileName):
    # 得到纯净语音、白噪声的地址
    # path_clean_test = "./datasets/TIMIT/test/clean/train/FAEM0_SI2022.WAV"
    path_noisy_test = fileName
    # 设置处理后语音文件的地址
    output_path_estimated_noisy_test = '/home/aone/PycharmProjects/kj-flask-system/static/enhanced/' + "MMSE" + fileName.split('/')[-1]

    # 读取音频文件，得到音频文件的数据和采样率
    # clean_test, sr = sf.read(path_clean_test)
    noisy_test, sr = sf.read(path_noisy_test)

    # 设置后验信噪比的min和max
    maxPosteriorSNR = 100
    minPosteriorSNR = 1

    # 参数初始化
    NFFT = 256  # 每个分析框由256个退化语音样本组成
    hop_length_sample = 128
    winfunc = 'hamming'

    # 设置平滑因子
    smoothFactorDD = 0.99  # 相当于一个权值

    # 语音信号的变化
    # noisy
    # 傅里叶变换=>将一个连续的信号（不方便处理）转换成一个个小信号的叠加（好处理）
    # 进行短时傅里叶变换=>将时域和频域相联系（傅里叶变换只反映出信号在频域的特性，无法捕捉到信号在时域信号上的不同）
    # STFT=>通过计算短重叠窗口上的离散傅里叶变换(将信号从时域变换到频域=>求出组成信号的正弦波的幅度和相位)来表示时频域中的信号。
    # 通过离散短时傅里叶变换进行频谱分解
    stft_noisy_test = librosa.stft(noisy_test, n_fft=NFFT, hop_length=hop_length_sample, window=winfunc)
    # 将短时傅里叶变换处理后的数据分离，得到噪声功率幅度和相位
    magnitude_noisy_test, phase_noisy_test = divide_magphase(stft_noisy_test, power=1)

    # 得到噪声功率谱（幅度的平方得到功率）
    pSpectrum = magnitude_noisy_test ** 2

    # 使用最小统计噪声功率谱来估计噪声方差
    estNoise = estnoisem(pSpectrum, hop_length_sample / sr)
    estNoise = estNoise

    # 得到每一帧的后验信噪比
    aPosterioriSNR = pSpectrum / estNoise
    aPosterioriSNR = aPosterioriSNR
    aPosterioriSNR[aPosterioriSNR > maxPosteriorSNR] = maxPosteriorSNR
    aPosterioriSNR[aPosterioriSNR < minPosteriorSNR] = minPosteriorSNR

    # 设置首帧先验信噪比，之后会用于记录前一帧的先验信噪比
    previousGainedaPosSNR = 1

    # 由之前的得到的噪声功率谱提取出采样点数和信号长度
    (nFrames, nFFT2) = pSpectrum.shape
    totalGain = []

    # 从含有噪声的语音信号的频谱分量中估计出纯净语音的频谱分量
    # 对语音信号一帧一帧处理
    for i in range(nFFT2):

        aPosterioriSNR_frame = aPosterioriSNR[:, i]

        # 第i-1帧的先验信噪比、后验信噪比，就可求出第i帧的先验信噪比
        # 这里取前一帧的后验信噪比
        oper = aPosterioriSNR_frame - 1
        oper[oper < 0] = 0
        # 计算第i帧的估计先验信噪比
        smoothed_a_priori_SNR = smoothFactorDD * previousGainedaPosSNR + (1 - smoothFactorDD) * oper

        # 最小均方误差估计
        # 第i帧的先验信噪比和后验信噪比得到后计算MMSE(因为增益函数仅仅取决于先验信噪比和后验信噪比，但是增益函数是独立于后验信噪比的)
        V = smoothed_a_priori_SNR * aPosterioriSNR_frame / (1 + smoothed_a_priori_SNR)

        # 计算由MMSE产生的增益函数
        # 性滤波系统的传递函数，在语音增强领域，通常也称为增益函数
        # 此处的增益函数是一个n*n数组
        # 得到MMSE幅度估计器的增益
        gain = smoothed_a_priori_SNR / (1 + smoothed_a_priori_SNR)
        # print(gain)

        if any(V < 1):
            gain[V < 1] = (math.gamma(1.5) * np.sqrt(V[V < 1])) / aPosterioriSNR_frame[V < 1] * np.exp(
                -1 * V[V < 1] / 2) * ((1 + V[V < 1]) * bessel(0, V[V < 1] / 2) + V[V < 1] * bessel(1, V[V < 1] / 2))

        previousGainedaPosSNR = (gain ** 2) * aPosterioriSNR_frame
        # 将多维矩阵转换为n*1数组
        totalGain.append(gain)
        # print(type(gain))
    # 将得到的gain矩阵转换为一维数组
    totalGain = np.array(totalGain)

    # 得到纯净语音信号的频谱分量
    # 利用之前得到的增益函数和噪声幅度谱，得到估计的纯净语音的幅度谱
    magnitude_estimated_clean = totalGain.T * magnitude_noisy_test
    # 进行短时傅里叶变换，由纯净语音的频谱分量和噪声相位（相位主值的最优估计量就是噪声相位本身）得到语音信号（时频信号）
    stft_reconstructed_clean = merge_magphase(magnitude_estimated_clean, phase_noisy_test)
    # 将得到的时频信号转换为声音信号
    signal_reconstructed_clean = librosa.istft(stft_reconstructed_clean, hop_length=hop_length_sample, window=winfunc)
    # 进行数据转换
    signal_reconstructed_clean = signal_reconstructed_clean.astype('float32')
    # 储存增强处理后的语音信号
    # wav.write(output_path_estimated_noisy_test, sr, signal_reconstructed_clean)
    sf.write(output_path_estimated_noisy_test,signal_reconstructed_clean,sr)

