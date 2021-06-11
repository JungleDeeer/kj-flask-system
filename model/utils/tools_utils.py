import logging
import importlib
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torch.nn as nn
import librosa
import numpy as np
from sklearn import cluster

def m_print(log):
    logging.info(log)
    print(log)


def initialize_config(module_cfg):
    """
    According to config items, load specific module dynamically with params.

    eg，config items as follow：
        module_cfg = {
            "module": "models.model",
            "main": "Model",
            "args": {...}
        }

    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def pad_to_longest(batch_data):
    """
    将dataset返回的batch数据按照最长的补齐
    :param batch_data:
    :return:
    """
    mixs, cleans, lengths, name = zip(*batch_data)
    mix_list = []
    clean_list = []

    for mix in mixs:
        mix_list.append(torch.Tensor(mix))
    for clean in cleans:
        clean_list.append(torch.Tensor(clean))

    mix_list = pad_sequence(mix_list).permute(1, 0)
    clean_list = pad_sequence(clean_list).permute(1, 0)

    return mix_list, clean_list, lengths, name

def pad_to_longest_label(batch_data):
    """
    将dataset返回的batch数据按照最长的补齐
    :param batch_data:
    :return:
    """
    mixs, cleans,label, lengths, name = zip(*batch_data)
    mix_list = []
    clean_list = []

    for mix in mixs:
        mix_list.append(torch.Tensor(mix))
    for clean in cleans:
        clean_list.append(torch.Tensor(clean))

    mix_list = pad_sequence(mix_list).permute(1, 0)
    clean_list = pad_sequence(clean_list).permute(1, 0)

    return mix_list, clean_list,label, lengths, name

def pad_to_longest_n(batch_data):
    """
    将dataset返回的batch数据按照最长的补齐
    :param batch_data:
    :return:
    """
    mixs, cleans,noises, lengths, name = zip(*batch_data)
    mix_list = []
    clean_list = []
    noise_list = []
    for mix in mixs:
        mix_list.append(torch.Tensor(mix))
    for clean in cleans:
        clean_list.append(torch.Tensor(clean))
    for noise in noises:
        noise_list.append(torch.Tensor(noise))
    mix_list = pad_sequence(mix_list).permute(1, 0)
    clean_list = pad_sequence(clean_list).permute(1, 0)
    noise_list = pad_sequence(noise_list).permute(1, 0)
    return mix_list, clean_list,noise_list, lengths, name

def print_networks(nets: list):
    """
    计算网络参数总量
    :param nets:
    :return:
    """
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network
    return params_of_all_networks

class ComplexMaskOnPolarCoo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sound_with_noise, unet_out):
        mag_mask, phase_mask = torchaudio.functional.magphase(unet_out)
        mag_input, phase_input = torchaudio.functional.magphase(sound_with_noise)
        mag_mask = torch.tanh(mag_mask)
        mag = mag_mask * mag_input
        phase = phase_mask + phase_input
        return torch.cat(((mag * torch.cos(phase)).unsqueeze(-1),
                          (mag * torch.sin(phase)).unsqueeze(-1)),dim=-1)


def noise_cluster(k , mfcc_feature):
    path = "../datasets/TIMIT/16k/noise/seen/"
    filter = {'babble', 'destroyerengine', 'destroyerops', 'factory1', 'resto'}
    mfccs = []
    for noise in filter:
        noise_path = path + noise + '.wav'
        if noise == 'resto':
            y, sr = librosa.load(noise_path, sr=None,dtype = "float32")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_feature)
            # print(mfcc.shape)
            mfcc = mfcc[:,:3676]
            mfccs.append(mfcc)
        else:
            y, sr = librosa.load(noise_path, sr=None,dtype ="float32")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_feature)
            mfccs.append(mfcc)
    mfccs = np.array(mfccs).reshape(-1,mfcc_feature)
    [centroid, label, inertia] = cluster.k_means(mfccs, k)
    return centroid

def SI_SDR(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
                      / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


import ctypes as ct


class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]


class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]


def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1
