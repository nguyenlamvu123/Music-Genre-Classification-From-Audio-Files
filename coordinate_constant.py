import json, librosa, os, time
from functools import wraps
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


path = os.path.dirname(__file__)
genr_ori = os.path.join(path, 'Data', 'genres_original')
genres = os.listdir(genr_ori)
##songname = os.path.join(genr_ori, genres[2], f'{genres[2]}.00029.wav')

header = 'filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo'
for i in range(1, 21):
    header += f' mfcc{i}_mean mfcc{i}_var'
header += ' label'
header = header.split()

num_mfcc = 20
sample_rate = 22050
n_fft = 2048
hop_length = 512
num_segment = 10
samples_per_segment = int(sample_rate*100/num_segment)  # số sample trong 100 giây chia ra thành num_segment=10 phần

debug = False
demo = False
##pre_dataset = True
epochs = 900 if not demo else 5

converter = LabelEncoder()
# model = tf.keras.models.load_model("model.keras") if os.path.exists("model.keras") else None


def timer(func):  # @timer
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if debug:
            print(f"Execution time of {func.__name__}: {end - start} seconds")
        return result
    return wrapper


def readfile(file="uid.txt", mod="r", cont=None, jso: bool = False):
    if not mod in ("w", "a", ):
        assert os.path.isfile(file), str(file)
    if mod == "r":
        with open(file, encoding="utf-8") as file:
            lines: list = file.readlines()
        return lines
    elif mod == "_r":
        with open(file, encoding="utf-8") as file:
            contents = file.read() if not jso else json.load(file)
        return contents
    elif mod == "rb":
        with open(file, mod) as file:
            contents = file.read()
        return contents
    elif mod in ("w", "a", ):
        with open(file, mod, encoding="utf-8") as fil_e:
            if not jso:
                fil_e.write(cont)
            else:
                json.dump(cont, fil_e, indent=2, ensure_ascii=False)


def argpa(lis=None):  # -> args
    import argparse
    parser = argparse.ArgumentParser()  # Initialize parser
    parser.add_argument(
        "duongdan", type=str, help="đường dẫn")

    args = parser.parse_args() if lis is None else parser.parse_args(lis)  # Read arguments from command line
    return args


def readdata(df):
    df = df.drop(labels="filename", axis=1)

    fit = StandardScaler()
    X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    return X, df


def extractfeature(y_seg, sr, fseg_name='_', g='_', my_3_csv=None):
    #Chromagram
    chroma_hop_length = 512
    chromagram = librosa.feature.chroma_stft(y=y_seg, sr=sample_rate, hop_length=chroma_hop_length)
    #Root Mean Square Energy
    RMSEn= librosa.feature.rms(y=y_seg)
    #Spectral Centroid
    spec_cent=librosa.feature.spectral_centroid(y=y_seg)
    #Spectral Bandwith
    spec_band=librosa.feature.spectral_bandwidth(y=y_seg,sr=sample_rate)
    #Rolloff
    spec_roll=librosa.feature.spectral_rolloff(y=y_seg,sr=sample_rate)
    #Zero Crossing Rate
    zero_crossing=librosa.feature.zero_crossing_rate(y=y_seg)
    #Harmonics and Perceptrual
    harmony, perceptr = librosa.effects.hpss(y=y_seg)
    #Tempo
    tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sample_rate)
    #MEDIAS Y VARIANZAS DE LOS MFCC
    mfcc=librosa.feature.mfcc(y=y_seg,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc=mfcc.T
    #algunos archivos difieren en len(mfcc) por 1. No cambia nada, saco el if que los descarta

    my_3_csv["chroma_stft_mean"].append(chromagram.mean())
    my_3_csv["chroma_stft_var"].append(chromagram.var())

    my_3_csv["rms_mean"].append(RMSEn.mean())
    my_3_csv["rms_var"].append(RMSEn.var())

    my_3_csv["spectral_centroid_mean"].append(spec_cent.mean())
    my_3_csv["spectral_centroid_var"].append(spec_cent.var())

    my_3_csv["spectral_bandwidth_mean"].append(spec_band.mean())
    my_3_csv["spectral_bandwidth_var"].append(spec_band.var())

    my_3_csv["rolloff_mean"].append(spec_roll.mean())
    my_3_csv["rolloff_var"].append(spec_roll.var())

    my_3_csv["zero_crossing_rate_mean"].append(zero_crossing.mean())
    my_3_csv["zero_crossing_rate_var"].append(zero_crossing.var())

    my_3_csv["harmony_mean"].append(harmony.mean())
    my_3_csv["harmony_var"].append(harmony.var())
    my_3_csv["perceptr_mean"].append(perceptr.mean())
    my_3_csv["perceptr_var"].append(perceptr.var())

    try:
        my_3_csv["tempo"].append(tempo[0])
    except TypeError:
        print('$$$$$$$$$$$$$$', tempo, 'in', fseg_name)
        my_3_csv["tempo"].append(tempo)

    my_3_csv["filename"].append(fseg_name)
    for x in range(20):
        feat1 = "mfcc" + str(x+1) + "_mean"
        feat2 = "mfcc" + str(x+1) + "_var"
        my_3_csv[feat1].append(mfcc[:,x].mean())
        my_3_csv[feat2].append(mfcc[:,x].var())
    return my_3_csv


def extractfeature_(y, sr, filename='_', g='_'):
    chroma_hop_length = 512  # 5000?
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop_length)
    RMSEn = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    # mfcc = mfcc.T
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    to_append = f'{filename} {chroma_stft.mean()} {chroma_stft.var()} {RMSEn.mean()} {RMSEn.var()} {spec_cent.mean()} {spec_cent.var()} {spec_bw.mean()} {spec_bw.var()} {rolloff.mean()} {rolloff.var()} {zcr.mean()} {zcr.var()} {harmony.mean()} {harmony.var()} {perceptr.mean()} {perceptr.var()} {tempo[0]}'
    for e in mfcc:
        to_append += f' {np.mean(e)} {np.var(e)}'
    # for x in range(20):
    #     to_append += f' {mfcc[:,x].mean()} {mfcc[:,x].var()}'
    to_append += f' {g}'
    return to_append


if __name__ == '__main__':
    df = pd.read_csv(f"{path}/try_running__features_3_sec.csv")  #  if os.path.exists(f"{path}/try_running__features_3_sec.csv") else None
    y, sr = librosa.load(songname, mono=True, duration=30)
    to_append = extractfeature(y, sr, filename='_', g='_')
