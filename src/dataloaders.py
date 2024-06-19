import pandas as pd
import numpy as np
import librosa
import json

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

def crear_rangos_transcripciones(df_annotations: pd.DataFrame, model_version : str, **kwargs):

    """
        Input:
            df_annotations: df de anotaciones del excel
            model_version: string versión del modelo usada en la definción de la pipeline
        Output:
            Diccionario con key de cada audio, teniendo cada uno de dos listas: rangos y texto
    """
    trans_path = kwargs.get('trans_path', 'data/TRANSCRIPCIONES/WHISPER')
    
    # Obtengo todos los audios diferents y defino un diccionario vacío
    audios_name = df_annotations['Audio_Name'].unique()
    trans_dict = {}

    # Loop para optener todas las tuplas de inicio fin de todas las transcripciones y guardarlas en el diccionario trans_dict
    for audio_name in audios_name:
        audio_name_json = audio_name[:-4] + '.json'

        with open(f'{trans_path}/{audio_name_json}') as f:
            audio_data = json.load(f)

        trans_dict[audio_name] = {}
        x = []
        y = []
        for segment in audio_data['segments']:
            x.append((segment['start'], segment['end']))
            y.append(segment['text'])

        trans_dict[audio_name]['rangos'] = x
        trans_dict[audio_name]['texto'] = y

    # Verificación de haber transcripto todos los audios que no se encuentran en development
    assert len(trans_dict.keys()) == 205, f'Deben de haber 205 audios con transcripciones, se transcribieron {len(trans_dict.keys())}'

    # Guardo el diccionario en archivo json
    with open(f'data/MODELS/{model_version}/transcripciones.json', 'w') as f: json.dump(trans_dict, f)

    return trans_dict

def crear_objetivos(df_annotations: pd.DataFrame, intervalos_transcripciones : dict, funcion_votacion, model_version: str, lag : float = 0, suavizado : bool = False):

    # Creo mi diccionario que voy a guardare como json
    targets_mean_vote = {}
    lista_audios = df_annotations['Audio_Name'].unique()

    # Loop por cada audio en la lista de audios que no estan en development
    for audio_name in lista_audios:

        print('Procesando el audio:', audio_name)

        df_audio = df_annotations[df_annotations['Audio_Name'] == audio_name]
        partes = df_audio['Part_Num'].max()

        # Inicializo valores para el audio en el diccionario
        targets_mean_vote[audio_name] = {}
        targets_mean_vote[audio_name]['rangos'] = []
        targets_mean_vote[audio_name]['targets'] = []
        targets_mean_vote[audio_name]['indice'] = []

        # Loop por cada parte del audio
        for part_num in range(partes):
            part_num = part_num + 1

            print(f'Procesando parte {part_num} de {audio_name}')

            # Obtengo votación para la parte del audio
            df_votacion = funcion_votacion(df_annotations, audio_name = audio_name, part_num = part_num, suavizado = suavizado)

            # Verificamos que solo halla un start time y nos lo quedamos
            assert len(df_audio[df_audio['Part_Num'] == part_num]['start_time'].unique()) == 1, 'Más de un start time'
            start_time = df_audio[df_audio['Part_Num'] == part_num]['start_time'].unique()[0]

            # Agregamos al audio el start time necesario
            df_votacion['Time'] = df_votacion['Time'] + start_time

            # Loop principal para cargar las votaciones promedio
            i = 0

            int_act = intervalos_transcripciones[audio_name]['rangos']
            ts_primera_anotacion = df_votacion['Time'][0]
            ts_ultima_anotacion = df_votacion['Time'][len(df_votacion) - 1]

            # Agrego padding de 0.1 segundos por lo visto en el análisis en la notebook 03
            while i + 1 <= len(int_act) and int_act[i][1] + 0.1 < ts_ultima_anotacion:

                if int_act[i][0] > ts_primera_anotacion:
                    inicio = int_act[i][0]
                    fin = int_act[i][1] + 0.1

                    votacion_promedio = df_votacion[(df_votacion['Time'] >= inicio + lag) & (df_votacion['Time'] <= fin + lag)][['Valence','Arousal','Dominance']].mean().values.tolist()

                    targets_mean_vote[audio_name]['targets'].append(votacion_promedio)
                    targets_mean_vote[audio_name]['rangos'].append((inicio, fin))
                    targets_mean_vote[audio_name]['indice'].append(i)

                i += 1

        assert len(targets_mean_vote[audio_name]['targets']) == len(targets_mean_vote[audio_name]['rangos']), 'Rangos no coinciden'

        print(f'{audio_name} procesado con éxito')
        print('')
        print('---------------------------------------------------------')

    # Verificación de haber transcripto todos los audios que no se encuentran en development
    assert len(targets_mean_vote.keys()) == 205, f'Deben de haber 205 audios con transcripciones, se procesaron {len(targets_mean_vote.keys())}'

    # Guardo el diccionario en archivo json
    with open(f'data/MODELS/{model_version}/objetivos.json', 'w') as f: json.dump(targets_mean_vote, f)

    return targets_mean_vote

def obtener_raw_data(df_annotations: pd.DataFrame, audio_name: str, dict_objetivos : dict, **kwargs):

    """
        Input:
            df_annotations: Dataframe excel de anotaciones
            audio_name: Nombre del audio, incluyendo el .wav
            dict_objetivos: Diccionario con los objetivos, debe tener los campos rango e indice
        
        Output:
            Dataframe con las columnas Data, Time, Indice y Audio_Name
    """
    data, time, sr = cargar_audio_data(df_annotations = df_annotations, audio_name = audio_name)

    assert len(data) == len(time), "La cantidad de data y la cantidad de tiempo no coinciden, revisar modulo de carga"

    df_audio = pd.DataFrame(data = data, columns = ['Data'])
    df_audio['Time'] = time

    x, y, z = [],[],[]
    for inter, ind in zip(dict_objetivos[audio_name]['rangos'], dict_objetivos[audio_name]['indice']):
        inicio = inter[0]
        fin = inter[1]
        x.append(tuple(df_audio[(df_audio['Time'] >= inicio) & (df_audio['Time'] <= fin)]['Data'].values.tolist()))
        y.append((inicio,fin))
        z.append(ind)

    df_audio = pd.DataFrame()
    df_audio['Data'] = x
    df_audio['Time'] = y
    df_audio['Indice'] = z
    df_audio['Audio_Name'] = audio_name

    return df_audio