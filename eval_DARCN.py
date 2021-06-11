import pystoi
import pypesq
import json
import argparse
import os
import glob
import soundfile as sf
from tqdm import tqdm
# from tools.utils import *
# from torch.utils.data import DataLoader
# from tools.metrics import *
import pandas as pd
import torch
from model.GRN.GRN import GRN
from model.CRN.CRN import CRN
from model.specsub.specsub import load_specsub
from model.wiener.wiener import load_wiener
from model.MMSE.MMSE import load_MMSE
from model.kalmen.kalmen import load_kalmen

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

import pystoi
import pypesq
"""
eval_DARCN 用于新的数据集16K_DARCN的测试部分
"""

# parser = argparse.ArgumentParser()
# parser.add_argument("-C", "--config", required=True, type=str,
#                     help="Specify the configuration file for testing (*.json). (same with enhance)")
# args = parser.parse_args()

def use_model(modelName,fileName):
    if modelName == "GRN":
        model = GRN()
        pth_path = "/home/aone/PycharmProjects/kj-flask-system/model/GRN/GRN_model.tar"
    elif modelName == "CRN":
        model = CRN()
        pth_path = "/home/aone/PycharmProjects/kj-flask-system/model/CRN/CRN_model.tar"
    model_checkpoint = torch.load(pth_path)
    model_static_dict = model_checkpoint["model"]
    # model_static_dict = model_checkpoint
    checkpoint_epoch = model_checkpoint['epoch']
    # checkpoint_epoch = 28
    print(f"Loading {model.__class__.__name__} checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.cuda()
    model.eval()

    '''validating'''
    with torch.no_grad():

        ori_mixs_wav, sr = sf.read(fileName, dtype="float32")
        length = ori_mixs_wav.shape[0]
        # for mixs_wav, cleans_wav, lengths, names in tqdm(dataloader):
        mixs_wav = torch.Tensor(ori_mixs_wav)
        mixs_wav = mixs_wav.unsqueeze(dim=0)
        mixs = torch.stft(mixs_wav,
                          n_fft=320,
                          hop_length=160,
                          win_length=320,
                          window=torch.hamming_window(320)).permute(0, 2, 1, 3).cuda()
        mixs_real = mixs[:, :, :, 0]
        mixs_imag = mixs[:, :, :, 1]
        mixs_mag = torch.sqrt(mixs_real ** 2 + mixs_imag ** 2)

        enhances_mag = model(mixs_mag)

        '''eval'''
        enhances_real = enhances_mag * mixs_real / mixs_mag
        enhances_imag = enhances_mag * mixs_imag / mixs_mag
        enhances = torch.stack([enhances_real, enhances_imag], 3)
        enhances = enhances.permute(0, 2, 1, 3)

        enhances_wav = torch.istft(enhances,
                                   n_fft=320,
                                   hop_length=160,
                                   win_length=320,
                                   window=torch.hamming_window(320).cuda(),
                                   length=length)

        enhances_wav = enhances_wav.squeeze().cpu().numpy()
        librosa.output.write_wav(
            '/home/aone/PycharmProjects/kj-flask-system/static/enhanced/' + modelName + fileName.split('/')[-1],
            enhances_wav.astype(np.float32), sr=sr)


def load_model(modelName, fileName):
    '''load model'''
    if modelName in ("GRN","CRN"):
        use_model(modelName,fileName)
    elif modelName == "specsub":
        load_specsub(fileName)
    elif modelName == "wiener":
        load_wiener(fileName)
    elif modelName == "MMSE":
        load_MMSE(fileName)
    elif modelName == "kalmen":
        load_kalmen(fileName)

    ori_mixs_wav, sr = sf.read(fileName, dtype="float32")
    enhances_wav, sr2 = sf.read('/home/aone/PycharmProjects/kj-flask-system/static/enhanced/' + modelName + fileName.split('/')[-1], dtype="float32")
    plt.clf()
    plt.subplot(221)
    librosa.display.waveplot(ori_mixs_wav, sr)
    plt.title('Mix Time Signal')

    plt.subplot(222)
    librosa.display.waveplot(enhances_wav, sr)
    plt.title(modelName+' Enhance Time Signal')

    plt.subplot(223)
    mixs = librosa.stft(ori_mixs_wav,
                        n_fft=320,
                        hop_length=160,
                        win_length=320)
    librosa.display.specshow(librosa.power_to_db(mixs), sr=sr, x_axis='time', y_axis='hz')
    plt.title('Mix Spectrum')

    plt.subplot(224)
    enhances = librosa.stft(enhances_wav,
                            n_fft=320,
                            hop_length=160,
                            win_length=320)
    librosa.display.specshow(librosa.power_to_db(enhances), sr=sr, x_axis='time', y_axis='hz')
    plt.title(modelName+' Enhance Spectrum')
    plt.tight_layout()
    plt.savefig(
        "/home/aone/PycharmProjects/kj-flask-system/static/spectrum/" + modelName + fileName.split('/')[-1].replace(
            '.WAV', '.jpg')
        .replace('.wav', '.jpg'))

    # plt.figure(figsize=(14, 5))
    # enhances_wav = enhances_wav.cpu().numpy()
    # for clean, enhance, length, name in zip(cleans_wav, enhances_wav, lengths, names):
    #     enhance = enhance[:length]
    # librosa.display.waveplot(enhance, sr=16000)
    # plt.title('enhance', fontsize=18)
    # plt.show()
    # exit()




def map_function(x):
    return x.split('+')[0]

if __name__ == '__main__':
    config = json.load(open(args.config))
    # main(config)
    df = pd.read_csv("output/noisy.csv", encoding='utf-8-sig')
    df = df.dropna()
    paths = df['path']
    print('type', 'stoi', 'pesq', 'si-sdr')
    print('total', np.mean(df['stoi']), np.mean(df['pesq']), np.mean(df['si-sdr']))
    SNRs = ['-5', '0', '5', '10']
    for SNR in SNRs:
        new_df = df
        new_df['path'] = df['path'].map(map_function)
        new_df = new_df[new_df['path']==SNR]
        print(SNR, np.mean(new_df['stoi']), np.mean(new_df['pesq']), np.mean(new_df['si-sdr']))
