import opensmile
import librosa
import numpy as np

## Extraído de la web de opensmile
def opensmile_features(data : np.ndarray):

    """
        Inputs:
            -data: Array de dos dimensiones con los features

        Output:
            Lista con los features
    """

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    x = []
    for i in data:
        x.append(list(smile.process_signal(i, sampling_rate = 16000).values[0]))

    return x

## Extraído de kaggle notebook (https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy/output)
# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate = rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = pitch_factor)

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y = data, frame_length = frame_length, hop_length = hop_length)
    return np.squeeze(rmse)

def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y = data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def traditional_features(data : np.ndarray, sr : int = 16000, frame_length=2048, hop_length=512):
    
    x = []
    for i in data:
        x.append(list(np.hstack((
                      zcr(i,frame_length,hop_length),
                      rmse(i,frame_length,hop_length),
                      mfcc(i,sr,frame_length,hop_length)
                        ))
                    )
                )

    return x