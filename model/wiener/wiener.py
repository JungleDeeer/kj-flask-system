# import wave
# from tools.utils import *
# import math
# import numpy as np
# # 打开WAV文档
# f = wave.open("./datasets/TIMIT/test/noisy/train/0+destroyerengine+FAEM0+FAEM0_SI2022.WAV")
# # 读取格式信息
# # (nchannels, sampwidth, framerate, nframes, comptype, compname)
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# fs = framerate
# # 读取波形数据
# str_data = f.readframes(nframes)
# f.close()
# # 将波形数据转换为数组
# noisy_speech = np.fromstring(str_data, dtype=np.short)
#
# mu= 0.98
# a_dd= 0.98
# eta= 0.15
# frame_dur= 20
# L= frame_dur* fs/ 1000
# hamming_win= np.hamming(L)
# U= (np.transpose(hamming_win)* hamming_win)/ L
#
#
# len_120ms= fs/ 1000* 120
#
# first_120ms= noisy_speech[1:len_120ms]
#
#  # =============now use Welch's method to estimate power spectrum with
# # % Hamming window and 50% overlap
# nsubframes= math.floor(len_120ms/ (L/ 2))- 1
# noise_ps= np.zeros( L, 1)
# n_start= 1
# for j in range(1, nsubframes):
#     noise= first_120ms[n_start: n_start+ L- 1]
#     noise= noise * hamming_win
#     noise_fft= np.fft.fft(noise, L)
#     noise_ps= noise_ps + (abs( noise_fft)*abs( noise_fft))/ (L* U)
#     n_start = n_start + L / 2
#
# noise_ps = noise_ps / nsubframes
#
# len1= L/ 2
# nframes= math.floor(len( noisy_speech)/ len1)- 1
# n_start= 1
#
# for j in range( 1, nframes):
#     noisy = noisy_speech[n_start: n_start + L - 1]
#     noisy = noisy * hamming_win
#     noisy_fft = np.fft.fft(noisy, L)
#     noisy_ps = (abs(noisy_fft)*abs( noisy_fft))/ (L * U)
#
#
#     if j == 1 :
#         posteri = noisy_ps/ noise_ps
#         posteri_prime = posteri - 1
#         posteri_prime[(posteri_prime < 0)] = 0
#         priori = a_dd + (1 - a_dd) * posteri_prime
#     else:
#         posteri = noisy_ps / noise_ps
#         posteri_prime = posteri - 1
#         posteri_prime[(posteri_prime < 0)] = 0
#         priori = a_dd * (G_prev*G_prev) * posteri_prev + (1 - a_dd) * posteri_prime
#
#
#     log_sigma_k = posteri * priori / (1 + priori) - np.log(1 + priori)
#     vad_decision(j) = sum(log_sigma_k) / L
#     if (vad_decision(j) < eta):
#         noise_ps = mu * noise_ps + (1 - mu) * noisy_ps
#         vad[n_start: n_start + L - 1]= 0
#
#     else:
#         vad[n_start: n_start + L - 1]= 1
#     G = np.sqrt(priori / (1 + priori))
#     enhanced = np.fft.ifft(noisy_fft * G, L)
#
#     if j == 1:
#         enhanced_speech[n_start: n_start + L / 2 - 1]= enhanced[1: L / 2]
#     else:
#         enhanced_speech[n_start: n_start + L / 2 - 1] = overlap + enhanced[1: L / 2]
#
#     overlap = enhanced[L / 2 + 1: L]
#     n_start = n_start + L / 2
#
#     G_prev = G
#     posteri_prev = posteri
#
import numpy as np
import wave
import librosa.display
#从噪声中提取信号的过滤和预测的方法，并以估计的结果与信号的真值间误差的最小均方值作为最佳准则
#利用因果的方法去逼近求解非因果的增益函数，主要方法有两大类：迭代的方法和非迭代的方法

# 输入纯净语音文件sp01.wav
# from traditional.show_spec import show_spec
# 输入带噪语音文件，此处可以选择不同的信噪比来查看效果


def load_wiener(fileName):
    f = wave.open(fileName)

    # 读取参数，需要用到的为声道数、量化位数、采样频率、采样点数
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    fs = framerate
    # 读取音频信息
    str_data = f.readframes(nframes)
    # 关闭文件
    f.close()

    # 将读取到的音频信息转换为数组
    x = np.fromstring(str_data, dtype=np.short)

    # 对带噪信号进行傅里叶变换
    x_FFT = abs(np.fft.fft(x))

    # 初始化参数
    len_ = 20 * fs // 1000  # 窗口大小
    PERC = 50  # 窗口重叠部分占比
    len1 = len_ * PERC // 100  # 重叠部分长度
    len2 = len_ - len1  # 非重叠部分长度

    # 设置默认参数
    Thres = 3  # VAD阈值
    Expnt = 2.0
    beta = 0.002  # 谱下限因子
    G = 0.9

    # 正弦窗
    i = np.linspace(0, len_ - 1, len_)
    win = np.sqrt(2 / (len_ + 1)) * np.sin(np.pi * (i + 1) / (len_ + 1))

    # 重叠部分标准化增益+50%重叠部分
    winGain = len2 / sum(win)

    # nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))转换为最接近二次幂
    nFFT = 2 * 2 ** 8
    # 噪声均值初始化分配
    noise_mean = np.zeros(nFFT)
    j = 1
    # 取带噪信号信号前五秒
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j: j + len_], nFFT))
        j = j + len_
    # 由于没有噪声信息，当前将带噪信号均值作为噪声
    noise_mu = abs(noise_mean / 5)

    # 初始化各种变量
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # === 开始处理 ==== #
    for n in range(0, Nframes):

        # 加窗
        insign = win * x[k - 1: k + len_ - 1]
        # 对每一帧进行傅里叶变换
        spec = np.fft.fft(insign, nFFT)
        # 计算带噪信号幅度
        sig = abs(spec)
        # 保存带噪信号相位信息
        theta = np.angle(spec)
        # 计算得到后验信噪比
        # np.linalg.norm矩阵范数反映了线性映射把一个向量映射为另一个向量，向量的“长度”缩放的比例。
        SNRpos = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        # --- 维纳滤波器 --- #

        # 对SNR较低的信号帧利用增益处理作更深度的噪声消除，而对SNR较高的信号帧则通过减小滤波系数增益以减低滤波器影响，从而降低噪声消除深度
        def berouti(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a

        def berouti1(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 3 - SNR * 2 / 20
            else:
                if SNR < -5.0:
                    a = 4
                if SNR > 20:
                    a = 1
            return a

        # 设置alpha过减因子(利用后验信噪比得到)
        if Expnt == 1.0:  # 幅度谱
            alpha = berouti1(SNRpos)
        else:  # 功率谱
            alpha = berouti(SNRpos)

        # 估计纯净语音:带噪语音功率谱-噪声功率谱
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
        # 纯信号小于噪声信号功率，消除估计纯净信号的负分量
        diffw = sub_speech - beta * noise_mu ** Expnt

        # beta 负部分
        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)
        if len(z) > 0:
            # 下限用估计的噪声信号表示
            for i in range(len(z)):
                sub_speech[z[i]] = beta * noise_mu[z[i]] ** Expnt

        # 计算先验信噪比
        # 先验信噪比=估计纯净语功率谱/噪声功率谱
        SNRpri = 10 * np.log10(np.linalg.norm(sub_speech ** (1 / Expnt), 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        # mel参数
        mel_max = 10
        mel_0 = (1 + 4 * mel_max) / 5
        s = 25 / (mel_max - 1)

        # 开始处理
        def get_mel(SNR):
            if -5.0 <= SNR <= 20.0:
                a = mel_0 - SNR / s
            else:
                if SNR < -5.0:
                    a = mel_max
                if SNR > 20:
                    a = 1
            return a

        # 设置mel
        mel = get_mel(SNRpri)

        # 增益函数Gk=估计纯净语音功率谱/估计纯净语音功率谱+噪声功率谱
        G_k = sub_speech / (sub_speech + mel * noise_mu ** Expnt)
        # 上式分子分母同时除以噪声功率得到基于先验信噪比的增益函数形式m/m+1,m=估计纯净语音功率谱/噪声信号功率谱
        # 利用增益函数滤波得到估计增强语音
        wf_speech = G_k * sig

        # --- 实现一个简单的VAD检测 --- #  ====》利用VAD（噪声活动性检测）在无语音帧计算得到的噪声功率谱估计
        if SNRpos < Thres:  # 更新噪声频谱图
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
            noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱

        # 加上相位
        # wf_speech[nFFT // 2 + 1:nFFT] = np.flipud(wf_speech[1:nFFT // 2])
        x_phase = wf_speech * np.exp(img * theta)

        # 进行离散傅里叶变换
        xi = np.fft.ifft(x_phase).real

        # --- 重叠相加 --- #
        xfinal[k - 1: k + len2 - 1] = x_old + xi[0: len1]
        x_old = xi[0 + len1: len_]
        k = k + len2

    # 保存
    wf = wave.open('/home/aone/PycharmProjects/kj-flask-system/static/enhanced/' + "wiener"+ fileName.split('/')[-1], 'wb')

    # 设置参数
    wf.setparams(params)
    # 将数组形式表示的信息恢复
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tostring())
    # 关闭文件
    wf.close()



# 输出波形图和频谱图


# show_spec('./data/sp01.wav', './data/in_SNR0_sp01.wav', './en_output/out_SNR0_sp01.wav', '(SNR=0dB)')
# show_spec('./data/sp01.wav', './data/in_SNR5_sp01.wav', './en_output/out_SNR5_sp01.wav', '(SNR=5dB)')
# show_spec('./data/sp01.wav', './data/in_SNR10_sp01.wav', './en_output/out_SNR10_sp01.wav', '(SNR=10dB)')
# show_spec('./data/sp01.wav', './data/in_SNR15_sp01.wav', './en_output/out_SNR15_sp01.wav', '(SNR=15dB)')

