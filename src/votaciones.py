import pandas as pd
import numpy as np

def votacion_promedio_simple(df_annotations: pd.DataFrame, part_num: int, pc_num: int = None, audio_name: str = None, suavizado : bool = False,**kwargs) -> pd.DataFrame:
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num (opcional): Part_Num del audio
            -audio_name (opcional): Audio_Name, incluyendo el .wav, si se da, ignora el pc_num

        Output:
            Pandas Dataframe de 3 columnas: Time, Valence, Arousal, Dominance.
            Cada columna representa la votación promedio para ese momento en el audio.
    """

    votation_means = pd.DataFrame(columns = ['Time','Vote','Emotion'])
    emotions = ['Valence','Arousal','Dominance']

    for emotion in emotions:
        
        time = pd.DataFrame(columns = ['Time','Annotation','Annotator'])

        if audio_name is not None: df_copy = df_annotations[(df_annotations['Audio_Name'] == audio_name) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        else: df_copy = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        
        for name, annotator, emot in zip(df_copy['Annotation_File'], df_copy['Annotator'], df_copy['Emotion']):
        
            temp_df = pd.read_csv(f'data/MSPCORPUS/Annotations/{emot}/{name}', skiprows=9, header=None, names=['Time', 'Annotation'])
            temp_df['Annotator'] = annotator
            time = pd.concat([time, temp_df], ignore_index = True)
            
        df_pivot = pd.DataFrame(time.pivot_table(columns = 'Annotator', index = 'Time', values = 'Annotation').to_records()).set_index('Time')
        df_pivot = df_pivot.fillna(method='ffill')
        df_pivot['Vote'] = df_pivot.mean(axis = 1)
        df_pivot['Emotion'] = emotion

        votation_means = pd.concat([votation_means, df_pivot.reset_index()])

    df_emotions_vote = pd.DataFrame(votation_means.pivot_table(columns = 'Emotion', index = 'Time', values = 'Vote').to_records()).set_index('Time')
    df_emotions_vote = df_emotions_vote.fillna(method='ffill')
    df_emotions_vote = df_emotions_vote.fillna(method='bfill') # NEW: Agregado para rellenar las anotacioens que faltan al empezar
    
    # Código para aplicación del suavizado por ventana movil, por ahora hard codeada a 300
    if suavizado:
        for emocion in ['Valence','Arousal','Dominance']:
            df_emotions_vote[emocion] = df_emotions_vote[emocion].rolling(int(len(df_emotions_vote[emocion])/300)).mean()
            df_emotions_vote = df_emotions_vote.fillna(method='bfill')
    
    return df_emotions_vote.reset_index()[['Time','Valence','Arousal','Dominance']]

def votacion_promedio_ponderada(df_annotations: pd.DataFrame, pesos_votacion: dict, part_num: int, pc_num: int = None, suavizado : bool = False, multiplicador: float = 1,**kwargs) -> pd.DataFrame:
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pesos: pesos usados para la votación
            -pc_num: PodCast Number
            -part_num (opcional): Part_Num del audio
            -audio_name (opcional): Audio_Name, incluyendo el .wav, si se da, ignora el pc_num
            -multiplicador (opcional): Se multiplica por el peso a agregar a un voto, sirve para incentivar
        Output:
            Pandas Dataframe de 4 columnas: Time, Valence, Arousal, Dominance.
            Cada columna representa la votación promedio para ese momento en el audio.
    """

    audio_name = kwargs.get('audio_name', None)
    
    votation_means = pd.DataFrame(columns = ['Time','Vote','Emotion'])
    emotions = ['Valence','Arousal','Dominance']

    for emotion in emotions:

        time = pd.DataFrame(columns = ['Time','Annotation','Annotator'])

        if audio_name is not None: df_copy = df_annotations[(df_annotations['Audio_Name'] == audio_name) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        else: df_copy = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        
        for name, annotator, emot in zip(df_copy['Annotation_File'], df_copy['Annotator'], df_copy['Emotion']):

            signo = np.sign(pesos_votacion[emotion][str(annotator)])
            temp_df = pd.read_csv(f'data/MSPCORPUS/Annotations/{emot}/{name}', skiprows=9, header=None, names=['Time', 'Annotation'])
            temp_df['Annotator'] = annotator
            temp_df['Corrector'] = np.where(temp_df['Annotation'] > 0, 1, -1)
            temp_df['Annotation_New'] = (abs(temp_df['Annotation']) * ((pesos_votacion[emotion][str(annotator)] * multiplicador) + signo)) * temp_df['Corrector']
            time = pd.concat([time, temp_df], ignore_index = True)

        df_pivot = pd.DataFrame(time.pivot_table(columns = 'Annotator', index = 'Time', values = 'Annotation_New').to_records()).set_index('Time')
        df_pivot = df_pivot.fillna(method='ffill')
        df_pivot['Vote'] = df_pivot.mean(axis = 1)
        df_pivot['Emotion'] = emotion

        votation_means = pd.concat([votation_means, df_pivot.reset_index()])

    df_emotions_vote = pd.DataFrame(votation_means.pivot_table(columns = 'Emotion', index = 'Time', values = 'Vote').to_records()).set_index('Time')
    df_emotions_vote = df_emotions_vote.fillna(method='ffill')

    # Código para aplicación del suavizado por ventana movil, por ahora hard codeada a 300
    if suavizado:
        for emocion in ['Valence','Arousal','Dominance']:
            df_emotions_vote[emocion] = df_emotions_vote[emocion].rolling(int(len(df_emotions_vote[emocion])/300)).mean()
            df_emotions_vote = df_emotions_vote.fillna(method='bfill')
            
    return df_emotions_vote.reset_index()[['Time','Valence','Arousal','Dominance']]