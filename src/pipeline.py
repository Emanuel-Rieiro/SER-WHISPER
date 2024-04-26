import numpy as np
import pandas as pd
import os
import json

from src.models import *
from src.dataloaders import crear_rangos_transcripciones, crear_objetivos, obtener_raw_data
from src.feature_extractors import opensmile_features
from src.votaciones import votacion_promedio_simple
from src.traductores import obtener_emocion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

class DataPipeline:
    def __init__(self, model_version : str, **kwargs):

        df_annotations = pd.read_excel('data/annotations.xlsx')
        df_annotations = df_annotations[df_annotations['Type'] != 'Development'].reset_index(drop = True)
        self.df_annotations = df_annotations
        self.model_version = model_version
        self.funcion_votacion = kwargs.get('funcion_votacion', votacion_promedio_simple)
        self.funcion_features = kwargs.get('funcion_features', opensmile_features)
        self.cache = kwargs.get('cache', False)
        self.mapping = kwargs.get('mapping', 'Ekman')

        # Crear directorios
        try: os.listdir(f'data/MODELS/{self.model_version}')
        except: os.mkdir(f'data/MODELS/{self.model_version}') 

        try: os.listdir(f'data/MODELS/{self.model_version}/FEATURES')
        except: os.mkdir(f'data/MODELS/{self.model_version}/FEATURES') 

    def crear_rangos_transcripciones(self):

        print('Iniciando proceso para crear rango de transcripciones')
        
        if not self.cache:
            self.transcripciones = crear_rangos_transcripciones(self.df_annotations, self.model_version)

        elif 'transcripciones.json' not in os.listdir(f'data/MODELS/{self.model_version}'):
            self.transcripciones = crear_rangos_transcripciones(self.df_annotations, self.model_version)
        
        else:
            with open(f'data/MODELS/{self.model_version}/transcripciones.json', 'r') as f: 
                self.transcripciones = json.load(f)

        print('Proceso para crear rango de transcripciones finalizado con éxito')
        print('---------------------------------------------------------')
    
    def crear_objetivos(self):

        print('Iniciando proceso para crear targets de entrenamiento')

        if not self.cache:

            self.dict_objetivos = crear_objetivos(self.df_annotations, 
                                      self.transcripciones, 
                                      self.funcion_votacion, 
                                      self.model_version)
            
        elif 'objetivos.json' not in os.listdir(f'data/MODELS/{self.model_version}'):

            self.dict_objetivos = crear_objetivos(self.df_annotations, 
                                                  self.transcripciones, 
                                                  self.funcion_votacion, 
                                                  self.model_version)
        
        else:

            with open(f'data/MODELS/{self.model_version}/objetivos.json', 'r') as f: self.dict_objetivos = json.load(f)

        print('Proceso para crear targets de entrenamiento finalizado con éxito')
        print('---------------------------------------------------------')

    def obtener_features(self):
        
        print('Iniciando proceso para obtener features')

        lista_audios = self.df_annotations['Audio_Name'].unique()
        self.df_features = pd.DataFrame()

        for audio_name in lista_audios:
            
            print('Procesando', audio_name)

            if not self.cache:

                df_audio = obtener_raw_data(self.df_annotations, audio_name, self.dict_objetivos, self.model_version)
                df_audio['Features'] = self.funcion_features(df_audio['Data'].values)

                df_audio.to_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv', index = False)
                self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

            elif f'{audio_name[:21]}.csv' not in os.listdir(f'data/MODELS/{self.model_version}/FEATURES'):
                
                df_audio = obtener_raw_data(self.df_annotations, audio_name, self.dict_objetivos, self.model_version)
                df_audio['Features'] = self.funcion_features(df_audio['Data'].values)
                
                df_audio.to_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv', index = False)
                self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

            else:
                
                print('Usando cache para', audio_name[:21])
                df_audio = pd.read_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv')
                df_audio['Features'] = self._convertir_strarray_a_array(df_audio['Features'])

                self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

            print(f'{audio_name} procesado con éxito')
            print('')
        
        print('Proceso para obtener features finalizado con éxito')
        print('---------------------------------------------------------')

    def acondicionar_dataset(self):
        
        print('Iniciando proceso para acondicionamiento')

        # Loop principal para obtener los tiempos y target en pandas
        df_ranges = pd.DataFrame()

        for _key in self.dict_objetivos.keys():
        
            X = []
            for segment, target, indice in zip(self.dict_objetivos[_key]['rangos'], self.dict_objetivos[_key]['targets'], self.dict_objetivos[_key]['indice']):
                x = []
                x.append(segment)
                x.append(target)
                x.append(indice)
                X.append(x)

            df = pd.DataFrame(X, columns = ['Time','Target','Indice'])
            df['Audio_Name'] = _key
            df_ranges = pd.concat([df_ranges, df], ignore_index = True)

        self.df_final = pd.merge(self.df_features, df_ranges, how = 'inner', left_on = ['Indice','Audio_Name'], right_on = ['Indice','Audio_Name'])

        print('Proceso para acondicionamiento finalizado')
        print('---------------------------------------------------------')

    def remover_duplicados(self):
        
        print('Iniciando proceso para remover duplicados')

        print('Antes', len(self.df_final))
        self.df_final['Duplicated'] = self.df_final['Indice'].astype(str) + self.df_final['Audio_Name']
        self.df_final = self.df_final.drop_duplicates(subset = 'Duplicated')
        self.df_final = self.df_final.drop('Duplicated', axis = 1)
        print('Despues', len(self.df_final))

        print('Proceso para remover duplicados finalizado')
        print('---------------------------------------------------------')

    def crear_target_categorico(self):
        
        print('Iniciando proceso para taget categorico')

        self.df_final['Target'] = [obtener_emocion(i[0],i[1],i[2], mapping = self.mapping) for i in self.df_final['Target']]

        print('Proceso para target categórico finalizado')
        print('---------------------------------------------------------')

    def print_distribucion_target_categorico(self):
        df_copy = self.df_final.copy()
        df_copy['Cuenta'] = 1
        df_copy.groupby('Target').count()[['Target','Cuenta']]

    def alinear_muestras(self, min_cuenta : int):
        df_copy = self.df_final.copy()
        df_copy['Cuenta'] = 1
        df_cuenta = df_copy.groupby('Target').count().reset_index()[['Target','Cuenta']]
        target_validos = df_cuenta[df_cuenta['Cuenta'] > min_cuenta]

        threshold = target_validos['Cuenta'].min()
        
        df_final_2 = pd.DataFrame()
        for target in target_validos['Target']:
            df_temp = self.df_final[self.df_final['Target'] == target].sample(threshold)
            df_final_2 = pd.concat([df_temp, df_final_2])

        self.df_final = df_final_2

    def entrenar(self):
        X = [i for i in self.df_final['Features'].values]
        Y = self.df_final['Target'].values

        # Encoder de las emociones
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

        # split de la data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)

        self.model = MyKerasModel(x_train.shape[1], len(self.df_final['Target'].unique()))
        self.model.compile_model()
        self.model.train_model(x_train, y_train, x_test, y_test)

    def _convertir_strarray_a_array(self, filas):
        
        x = []
        for fila in filas:
            fila = fila.replace('[','').replace(']','').split(',')
            x.append([float(i) for i in fila])
        
        return x

    def run_pipeline(self, steps):
        for step in steps: step()