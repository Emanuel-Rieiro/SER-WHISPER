import os
import pandas as pd
import numpy as np

def suavizado_normalizado(df, window_size, step, column_name, annotator_number):

  i = step
  x, y = [], []

  while i < df['Time'].max():
    x.append(np.median(df[(df['Time'] >= i - window_size) & (df['Time'] <= i)][f'Annotator_{annotator_number}']))
    y.append(i)
    i += step

  x.append(np.median(df[(df['Time'] >= i - window_size) & (df['Time'] <= i)][f'Annotator_{annotator_number}']))
  y.append(i)

  return pd.DataFrame({'Time': y, column_name: x}).fillna(method='bfill')

def main():

    # Definir el directorio donde están los archivos CSV en Google Drive
    emotions = ['Arousal','Dominance','Valence']

    for emotion in emotions:

        directory = f'../data/MSPCORPUS/Annotations/{emotion}'
        files = os.listdir(f'../data/ANNOTATIONS-POST/{emotion}')

        # Leer cada archivo CSV en el directorio
        for filename in os.listdir(directory):
            if filename.endswith('.csv') and f'{filename}' not in files:
                print(f'{emotion}, {filename}')

                filepath = os.path.join(directory, filename)

                # Extraer la información del nombre del archivo
                parts = filename.split('_')
                annotator_number = parts[3].split('.')[0]

                # Leer el archivo CSV ignorando las primeras 9 líneas
                df = pd.read_csv(filepath, skiprows=9, header=None)
                df.columns = ['Time', f'Annotator_{annotator_number}']

                df = suavizado_normalizado(df, window_size = 0.5, step = 1 / 59, column_name = f'Annotator_{annotator_number}', annotator_number = annotator_number)

                df.to_csv(f'../data/ANNOTATIONS-POST/{emotion}/{filename}', index=False)

if __name__ == '__main__':

    main()