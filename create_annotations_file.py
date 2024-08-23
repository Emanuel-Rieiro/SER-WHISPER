import librosa
import os
import pandas as pd
import numpy as np


def main():

    # Lee el archivo que identifica como los audios estan divididos
    df_conv = pd.read_csv(f"data/MSPCORPUS/Time_Labels/conversation_parts.txt", delimiter=";", header=None, names=['Conversation_Part', 'start_time', 'end_time'])

    # Creamos dos columnas con el podcast number y la parte de ese podcast
    df_conv['PC_Num'] = df_conv['Conversation_Part'].apply(lambda x: x[17:21]).astype(int)
    df_conv['Part_Num'] = df_conv['Conversation_Part'].apply(lambda x: x[22:23]).astype(int)

    mem = {}

    def add_sync_time_columns(row):
        if row.Part_Num == 1:
            st = 0
            mem[row.PC_Num] = row.start_time
        else:
            st = row.start_time - mem[row.PC_Num]

        row['m_start_time'] = st
        row['m_end_time'] = row.end_time - mem[row.PC_Num]

        return row

    # Una fila con los audios en formato inicio: 0 y final: final - inicio
    df_conv = df_conv.apply(lambda row: add_sync_time_columns(row), axis=1)
    df_conv = df_conv[['Conversation_Part', 'm_start_time', 'm_end_time', 'PC_Num', 'Part_Num']]
    df_conv['Audio_Name'] = df_conv['Conversation_Part'].apply(lambda x: x[0:21]) + ".wav"
    df_conv = df_conv.rename({'m_start_time':'start_time','m_end_time':'end_time'}, axis = 1)


    # Obtener anotadores y emoción por parte
    emociones = ['Arousal','Dominance','Valence']
    X = []

    for emocion in emociones:

        for file in os.listdir(f'data/MSPCORPUS/Annotations/{emocion}'):

            conv_part = file[:-8]
            emotion = emocion
            annotator = file[-7:-4]
            annotation_file = file

            x = []
            x.append(conv_part)
            x.append(emotion)
            x.append(annotator)
            x.append(annotation_file)

            X.append(x)

    # Guardamos resultado en un dataframe
    df_expand = pd.DataFrame(X, columns = ['Conversation_Part','Emotion','Annotator','Annotation_File'])

    # Juntamos el dataframe expandido con el base
    df_annotations = pd.merge(df_conv, df_expand, how = 'left', on = 'Conversation_Part')

    # Cargamos archivo de texto con los tipos
    with open('data/MSPCORPUS/partitions.txt') as f:
        txt_file = f.readlines()

    list_types = [i.split(';') for i in txt_file]
    df_types = pd.DataFrame(list_types, columns = ['Audio_Name','Type'])

    # Formato para merge
    df_types['Type'] = df_types['Type'].str.replace('\n','')
    df_types['Audio_Name'] = df_types['Audio_Name'] + '.wav'

    df_annotations = pd.merge(df_annotations, df_types, how = 'left', on = 'Audio_Name')
    df_annotations = df_annotations[['Audio_Name','Conversation_Part','Annotation_File','Emotion','Annotator','PC_Num','Part_Num','Type','start_time','end_time']]
    df_annotations.to_excel('data/annotations.xlsx', index = False)

if __name__ == '__main__':

    main()