import json, librosa, os, time
from functools import wraps
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


path = "/home/bhm-ai/music_classification"
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
songname = f'{path}/Data/genres_original/jazz/jazz.00029.wav'

header = 'filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo'
for i in range(1, 21):
    header += f' mfcc{i}_mean mfcc{i}_var'
header += ' label'
header = header.split()

num_mfcc = 20
n_fft = 2048
hop_length = 512

debug = False
demo = True
pre_dataset = True
epochs = 900 if not demo else 5

converter = LabelEncoder()
model = tf.keras.models.load_model("model.keras")
df = pd.read_csv(f"{path}/try_running__features_3_sec.csv")
class_encod = df.iloc[:,-1]
Y = converter.fit_transform(class_encod)


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


def readdata(df):
    df = df.drop(labels="filename", axis=1)

    fit = StandardScaler()
    X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    return X, df


def extractfeature(y, sr, filename='_', g='_'):
    chroma_hop_length = 512  # 5000?
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop_length)
    RMSEn = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    to_append = f'{filename} {chroma_stft.mean()} {chroma_stft.var()} {RMSEn.mean()} {RMSEn.var()} {spec_cent.mean()} {spec_cent.var()} {spec_bw.mean()} {spec_bw.var()} {rolloff.mean()} {rolloff.var()} {zcr.mean()} {zcr.var()} {harmony.mean()} {harmony.var()} {perceptr.mean()} {perceptr.var()} {tempo}'
    for e in mfcc:
        to_append += f' {np.mean(e)} {np.var(e)}'
    # for x in range(20):
    #     to_append += f' {mfcc[:,x].mean()} {mfcc[:,x].var()}'
    to_append += f' {g}'
    return to_append


if __name__ == '__main__':
    pass
