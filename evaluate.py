import argparse
import auraloss
import functools
import json
import librosa
import math
import numpy as np
import os
import pysptk
import pyworld
import torch
import torchaudio as ta


from pathlib import Path

#from cargan.evaluate.objective.metrics import Pitch
#from cargan.preprocess.pitch import from_audio
#from fastdtw import fastdtw
from pesq import pesq
from scipy.io.wavfile import read
from scipy.spatial.distance import euclidean
from tqdm import tqdm

SR_TARGET = 24000
MAX_WAV_VALUE = 32768.0



class Harvest:
    def __init__(self, sample_rate, hop_size, f0_min=50, f0_max=1000):
        self.fs = sample_rate
        self.frame_period = hop_size * 1000 / sample_rate
        self.f0_floor = float(f0_min)
        self.f0_ceil = float(f0_max)

    def __call__(self, x):
        # convert to numpy
        x = x.double().numpy()
        f0, _ = pyworld.harvest(x, fs=self.fs, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, 
                                frame_period=self.frame_period)
        # convert to f32
        f0 = f0.astype(np.float32)

        f0 = torch.from_numpy(f0)
        return f0


class ScalarMetric:
    def __init__(self):
        self.num_samples = 0
        self.sum = 0.0

    def update(self, sample):
        assert(isinstance(sample, (float, int)))

        self.sum += sample 
        self.num_samples += 1
    
    @property
    def average(self):
        return self.sum / self.num_samples



class MCDMetric:
    def __init__(self):
        self.num_frames = 0
        self.sum_of_dists = 0

    def update(self, ori, syn):
        assert(ori.shape[0] == syn.shape[0])

        dist = (ori - syn).pow(2).sum(dim=-1).sqrt().sum().item()
        self.sum_of_dists += dist 
        self.num_frames += ori.shape[0]
    
    @property
    def mcd(self):
        return 10.0 / math.log(10.0) * math.sqrt(2.0) * self.sum_of_dists / self.num_frames

        # 'MCD': 10.0 / np.log(10.0) * np.sqrt(2.0) * float(s) / float(frames_tot),

class PitchMetric:
    def __init__(self, log=True):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.sse = 0.0
        self.log = log

    
    def update(self, ref_f0, syn_f0):
        # check length
        if ref_f0.shape[0] != syn_f0.shape[0]:
            raise ValueError(f"wrong length")
        
        # get voiced frame
        ref_v_mask = ref_f0 > 0
        syn_v_mask = syn_f0 > 0

        # calculate the tp, fp, tn, fn
        self.tp += (ref_v_mask & syn_v_mask).sum().item()
        self.fp += (~ref_v_mask & syn_v_mask).sum().item()
        self.tn += (~ref_v_mask & ~syn_v_mask).sum().item()
        self.fn += (ref_v_mask & syn_v_mask).sum().item()

        # select the true positive and calculate the sse
        tp_mask = (ref_v_mask & syn_v_mask)
        ref_f0 = ref_f0[tp_mask]
        syn_f0 = syn_f0[tp_mask]

        if self.log:
            self.sse += ( ref_f0.log() - syn_f0.log() ).pow(2).sum().item()
        else:
            self.sse += ( ref_f0 - syn_f0 ).pow(2).sum().item()
    
    @property
    def rmse(self):
        return math.sqrt(self.sse / self.tp)

    @property
    def f1_score(self):
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) 

    @property
    def precision(self):
        return self.tp/(self.tp + self.fp)

    @property
    def recall(self):
        return self.tp/(self.tp + self.fn)



def load_wav(full_path):
    wav, sampling_rate = ta.load(full_path)

    if sampling_rate != SR_TARGET:
        raise IOError(
            f'Sampling rate of the file {full_path} is {sampling_rate} Hz, but the model requires {SR_TARGET} Hz'
        )

    #audio = audio / MAX_WAV_VALUE

    #audio = torch.FloatTensor(audio)
    #audio = audio.unsqueeze(0)
    
    wav = wav[0]

    return wav


def readmgc(x):
    is_tensor = torch.is_tensor(x)
    
    if is_tensor:
        x = x.numpy()

    frame_length = 1024
    hop_length = 256
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    
    if is_tensor:
        mgc = torch.from_numpy(mgc)

    return mgc


def trim(*tensors):
    min_len = min([x.shape[0] for x in tensors])
    
    return [ x[:min_len] for x in tensors ]



def evaluate_new(wav_dir, keys=[0.0]):
    # 個別指標 
    # - STFT: 原本的音檔 和 原音高的音檔
    # - MGC:  原本的音檔  和 原音高的音檔
    # - PESQ:  原本的音檔 和 原音高的音檔
    # - F0[+/-]\d: 原本的音檔和 音高 ±n 的音檔 
    # - UV[+/-]\d
    
    # 找目標音檔和對應的 semi-tones

    files = list(wav_dir.glob("**/*_ori.wav"))

    # 參數抽取函數
    
    resampler_16k = ta.transforms.Resample(SR_TARGET, 16000)
    resampler_22k = ta.transforms.Resample(SR_TARGET, 22050)
    harvest = Harvest(SR_TARGET, hop_size=120, f0_min=60, f0_max=1000)
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    
    # 統計的指標
    stft_metric = ScalarMetric()
    mcd_metric = MCDMetric()
    pesq_metric = ScalarMetric()
    pitch_metrics = { key: PitchMetric(log=True) for key in keys }
    

    with torch.inference_mode():
        iterator = tqdm(files, dynamic_ncols=True, desc=f'Evaluating {wav_dir}')
        # 先抽取原音檔
        for ori_path in iterator:
            # 讀取 wav
            x = load_wav(ori_path)
            f0_x = harvest(x)
            

            for key in keys:
                factor = 2**(key/12)
                filename = ori_path.stem.replace("_ori", f"_{factor:.2f}") + ".wav"
                syn_path = ori_path.parent / filename
                
                # 讀取合成的音檔
                y = load_wav(syn_path)
                f0_y = harvest(y)
                
                # 確保長度相同
                _f0_x, _f0_y = trim(f0_x, f0_y)

                pitch_metrics[key].update(_f0_x * factor, _f0_y)


                if key == 0:
                    # MRSTFT calculation
                    _x, _y = trim(x, y)
                    loss = loss_mrstft(_y.view(1, 1, -1), _x.view(1, 1, -1)).item()
                    stft_metric.update(loss)

                    # PESQ
                    x_16k = resampler_16k(x)
                    y_16k = resampler_16k(y)
                    x_16k_i16 = (x_16k * MAX_WAV_VALUE).short()
                    y_16k_i16 = (y_16k * MAX_WAV_VALUE).short()
                    
                    _x, _y = trim(x_16k_i16, y_16k_i16)

                    loss = pesq(16000, x_16k_i16.numpy(), y_16k_i16.numpy(), 'wb')
                    pesq_metric.update(loss)

                    # MCD calculation
                    x_22k = resampler_22k(x)
                    y_22k = resampler_22k(y)
                    x_22k_f64 = (x_22k * MAX_WAV_VALUE).double()
                    y_22k_f64 = (y_22k * MAX_WAV_VALUE).double()
                    
                    _x, _y = trim(x_22k_f64, y_22k_f64)

                    x_mgc = readmgc(_x)
                    y_mgc = readmgc(_y)
                    
                    mcd_metric.update(x_mgc, y_mgc)

    result = {
        'STFT': stft_metric.average, 
        'MCD': mcd_metric.mcd,
        'PESQ': pesq_metric.average,
        'RMSE(LF0)': {
            f"{key:+d}": pitch_metrics[key].rmse  
            for key in keys 
        },
        'F1Score(UV)': {
            f"{key:+d}": pitch_metrics[key].f1_score
            for key in keys
        }
    }


    return result


# 下面的要改成單一的資料夾
# 找出下面所有 _ori.wav, 和對應的目標

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='+', type=Path, help="輸入的資料夾。")
    parser.add_argument('--keys', nargs='+', type=float, required=True, help="輸入多個浮點數值")
    parser.add_argument('--output_file', default=None)

    args = parser.parse_args()
    keys = list(set(args.keys))
    
    results = {}

    for wav_dir in set(args.dirs):
        result = evaluate_new(wav_dir, args.keys)    
        results.append(result)
    
    print(results)

    if a.output_file:
        # Write results
        with open(args.output_file, 'w') as file:
            json.dump(results, fp=file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
