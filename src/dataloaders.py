import pandas as pd
import numpy as np
import librosa

def cargar_audio_data(df_annotations: pd.DataFrame, pc_num: int = None, audio_name: str = None) -> (np.ndarray, int):
    
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -part_num: Audio Part
            -pc_num (optional): PodCast Number
            -audio_name (optional): Audio name, including the .wav extension (Ex: MSP-Conversation_0002.wav)

        Output:
            1- A numpy array with the audio time series
            2- Integer sampling rate
    """
    
    if audio_name is not None: df_part_data = df_annotations[df_annotations['Audio_Name'] == audio_name].reset_index()
    else: df_part_data = df_annotations[df_annotations['PC_Num'] == pc_num].reset_index()

    audio_name = df_part_data['Audio_Name'][0]
    audio_path = "data/MSPCORPUS/Audio/" + audio_name

    data, sr = librosa.load(audio_path, sr = None)

    time = np.arange(0, len(data)) * (1.0 / sr)

    return data, time, sr