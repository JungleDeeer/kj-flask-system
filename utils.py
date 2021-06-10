import torch
import numpy as np
from GRN import GRN
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display
model = None
#读取模型函数
def load_model(modelname,filename):


    global model
    if modelname == "GRN":
        model = GRN()
    # path = "./model/" + modelname + "/GRN_model.tar"
    path = "/home/aone/PycharmProjects/kj-flask-system/model/GRN/GRN_model.tar"
    model_checkpoint = torch.load(
        path)
    # print(model_checkpoint)
    model_static_dict = model_checkpoint["model"]
    model.load_state_dict(model_static_dict)
    model.cuda()
    model.eval()

    mixs,mixs_mag,mixs_real,mixs_imag = audio_process(filename)
    enhance_mag = model(mixs_mag)
    mag2wav(enhance_mag,filename,mixs_real,mixs_mag,mixs_imag,mixs)



'''
    数据处理函数,把wav文件处理成幅度谱,之后放进模型 
    mixs_mag = audio_process(wav)
    enhance = model(mixs_mag)
    得到增强语音,需要转换成wav输出哈
'''



def audio_process(mixs_wav):

    #计算幅度谱
    win_len = 256
    hop_len = 128
    # print(type(mixs_wav))
    mix, sr = sf.read(mixs_wav, dtype="float32")
    mixs_wav = np.frombuffer(mix, dtype=np.float32)
    mixs_wav = torch.from_numpy(mixs_wav)
    # print(type(hop_len),type(mixs_wav))
    window = torch.hamming_window(win_len)
    # mixs_wav = torch.cuda.FloatTensor(mixs_wav)
    with torch.no_grad():
        #傅立叶变换
        mixs = torch.stft(mixs_wav,
                      n_fft=np.int(2 ** np.ceil(np.log2(win_len))),
                      hop_length=hop_len,
                      win_length=win_len,
                      window=window,
                          )
        print(type(mixs.numpy()))
        #得到mixs的实数谱和虚数谱
        mixs1 = mixs.unsqueeze(dim=0)
        mixs1 = mixs1.permute( 0,2, 1,3).cuda()
        mixs_real = mixs1[:, :, :, 0]
        mixs_imag = mixs1[:, :, :, 1]
        #得到幅度谱
        mixs_mag = torch.sqrt(mixs_real ** 2 + mixs_imag ** 2)
        mixs = librosa.stft(mix, n_fft=np.int(2 ** np.ceil(np.log2(win_len))),
                      hop_length=hop_len,
                      win_length=win_len,)
    return mixs,mixs_mag,mixs_real,mixs_imag

def mag2wav(enhances_mag,mixs_wav,mixs_real,mixs_mag,mixs_imag,mixs):

    plt.figure(figsize=(15, 10))

    win_len = 256
    hop_len = 128

    mix, sr = sf.read(mixs_wav, dtype="float32")
    length = mix.shape[0]

    enhances_real = enhances_mag * mixs_real / mixs_mag
    enhances_imag = enhances_mag * mixs_imag / mixs_mag
    enhances = torch.stack([enhances_real, enhances_imag], 3)
    enhances = enhances.permute(0, 2, 1, 3)
    enhances.squeeze(dim=0)
    #反傅立叶变换
    enhances_wav = torch.istft(enhances,
                               n_fft=np.int(2 ** np.ceil(np.log2(win_len))),
                               hop_length=hop_len,
                               win_length=win_len,
                               window=torch.hamming_window(win_len).cuda(),
                               length=length)
    # enhancer.initialize(opt, model, dataloader, noise, snr, enhancement_dir)
    #     print(enhances_wav.shape)
    enhances_wav = enhances_wav.cpu().detach().numpy()
    enhance = enhances_wav.squeeze()
    enhance_spec = librosa.stft(enhance, n_fft=np.int(2 ** np.ceil(np.log2(win_len))),
                        hop_length=hop_len,
                        win_length=win_len, )
    # sf.write("D:/Chrome downloads/kj-flask-system/kj-flask-system/date/enhanced/" + mixs_wav.split('/')[-1], enhance, samplerate=sr)
    # sf.write("D:/Chrome downloads/kj-flask-system/kj-flask-system/date/new.WAV"
    #          , enhance, samplerate=sr)
    librosa.output.write_wav('/home/aone/PycharmProjects/kj-flask-system/static/enhanced/'+ mixs_wav.split('/')[-1], enhance.astype(np.float32), sr=sr)

    plt.subplot(221)
    librosa.display.waveplot(mix, sr)
    plt.title('Mix Time Signal')

    plt.subplot(222)
    librosa.display.waveplot(enhance, sr)
    plt.title('Enhance Time Signal')

    plt.subplot(223)
    # print(mixs.numpy())
    # mixs = np.reshape(mixs.numpy(),(-1,2))
    # print("################################################")
    # print(mixs)
    # plt.plot(mixs)
    librosa.display.specshow(librosa.power_to_db(mixs), sr=sr, x_axis='time', y_axis='hz')
    plt.title('Mix Spectrum')



    plt.subplot(224)
    # print(enhances)
    enhances = np.reshape(enhances.cpu().detach().numpy(),(-1,2))
    # print("#########################")
    # print(enhances)
    # plt.plot(enhances)
    librosa.display.specshow(librosa.power_to_db(enhance_spec), sr=sr, x_axis='time', y_axis='hz')
    plt.title('Enhance Spectrum')
    plt.savefig("/home/aone/PycharmProjects/kj-flask-system/static/spectrum/"+ mixs_wav.split('/')[-1].replace('.WAV','.jpg').replace('.wav','.jpg'))
    # plt.show()

    return enhance

if __name__ == '__main__':
    # print(torch.__version__)

    # load_model("GRN","./data/0babbleFDHC0FDHC0_SI929.WAV")
    load_model("GRN", "/home/aone/PycharmProjects/kj-flask-system/data/0babbleFDHC0FDHC0_SI929.WAV")

