import pandas as pd

def votacion_promedio_simple(df_annotations: pd.DataFrame, part_num: int, pc_num: int = None, audio_name: str = None) -> pd.DataFrame:
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 
            -audio_name (optional): Audio name, including the .wav extension 

        Output:
            Dataframe consisting of 3 columns: Time, Arousal, Dominance, Valence.
            The Arousal, Dominance and Valence columns represent the mean vote calculated for corresponding Time in the audio.
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

    return df_emotions_vote.reset_index()