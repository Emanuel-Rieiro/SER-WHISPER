

# Necesario para hacer andar esta funcionalidad de sklearn
import sys
import joblib

import numpy as np
import pandas as pd
import os
import json

import logging
import pickle
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
        self.min_muestras = kwargs.get('min_muestras', 400)
        self.filemode = kwargs.get('filemode', 'w')
        self.encoder = None
        self.scaler = None

        # Crear directorios
        try: os.listdir(f'data/MODELS')
        except: os.mkdir(f'data/MODELS')

        try: os.listdir(f'data/MODELS/{self.model_version}')
        except: os.mkdir(f'data/MODELS/{self.model_version}')

        try: os.listdir(f'data/MODELS/{self.model_version}/FEATURES')
        except: os.mkdir(f'data/MODELS/{self.model_version}/FEATURES')

        # Logging
        logging.basicConfig(filename = f"data/MODELS/{self.model_version}/std.log", 
					        format = '%(asctime)s %(message)s', 
					        filemode = self.filemode,
                            level = logging.NOTSET)
        
        self.logger = logging.getLogger() 

        # Imprimir detalles de la build
        self._imprimir_detalles()

    def _print(self, string_param : str):
        print(string_param)
        self.logger.info(string_param)

    def _imprimir_detalles(self):

        self._print(f'Parametros del modelo')
        self._print(f'model_version: {self.model_version}')
        self._print(f'cache: {str(self.cache)}')
        self._print(f'funcion_features: {self.funcion_features.__name__}')
        self._print('')
        self._print('---------------------------------------------------------')

    def crear_rangos_transcripciones(self):
        
        """
            Output: Diccionario de keys con el nombre del audio, incluyendo el .wav y arrays "texto" y "rangos"
        """

        self._print('Iniciando proceso para crear rango de transcripciones')
        
        if not self.cache:
            self.transcripciones = crear_rangos_transcripciones(self.df_annotations, self.model_version)

        elif 'transcripciones.json' not in os.listdir(f'data/MODELS/{self.model_version}'):
            self.transcripciones = crear_rangos_transcripciones(self.df_annotations, self.model_version)
        
        else:
            with open(f'data/MODELS/{self.model_version}/transcripciones.json', 'r') as f: 
                self.transcripciones = json.load(f)

        self._print('Proceso para crear rango de transcripciones finalizado con éxito')
        self._print('---------------------------------------------------------')
    
    def crear_objetivos(self):

        self._print('Iniciando proceso para crear targets de entrenamiento')

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

        self._print('Proceso para crear targets de entrenamiento finalizado con éxito')
        self._print('---------------------------------------------------------')

    def obtener_features(self):
        
        self._print('Iniciando proceso para obtener features')

        lista_audios = self.df_annotations['Audio_Name'].unique()
        self.df_features = pd.DataFrame()
        step_skip = False

        # Intentamos ver si ya tenemos el archivo final del proceso entero
        if self.cache and os.path.exists(f'data/MODELS/{self.model_version}/FEATURES/df_features.csv'):
            
            self._print('Se ha encontrado el dataset de features ya generado, cargando...')
            self.df_features = pd.read_csv(f'data/MODELS/{self.model_version}/FEATURES/df_features.csv')
            self.df_features['Features'] = self._convertir_strarray_a_array(self.df_features['Features'])
            self._print('Dataset de features cargado con éxito')
        
        # Loop principal para generar el dataset de features
        else:

            for audio_name in lista_audios:

                self._print(f'Procesando {audio_name}')

                # Caso 1, no hay cache, por lo cual se ejecuta todo desde 0
                if not self.cache:

                    df_audio = obtener_raw_data(self.df_annotations, audio_name, self.dict_objetivos, self.model_version)
                    df_audio['Features'] = self.funcion_features(df_audio['Data'].values)

                    df_audio.to_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv', index = False)
                    self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

                # Caso 2, hay cache, pero el audio no se encuentra procesado
                elif f'{audio_name[:21]}.csv' not in os.listdir(f'data/MODELS/{self.model_version}/FEATURES'):

                    df_audio = obtener_raw_data(self.df_annotations, audio_name, self.dict_objetivos, self.model_version)
                    df_audio['Features'] = self.funcion_features(df_audio['Data'].values)

                    df_audio.to_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv', index = False)
                    self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

                # Caso 3, Hay cache y el audio se encuentra procesado
                else:

                    self._print(f'Usando cache para {audio_name[:21]}')
                    df_audio = pd.read_csv(f'data/MODELS/{self.model_version}/FEATURES/{audio_name[:21]}.csv')
                    df_audio['Features'] = self._convertir_strarray_a_array(df_audio['Features'])

                    self.df_features = pd.concat([df_audio[['Audio_Name','Indice','Features']], self.df_features])

                self._print(f'{audio_name} procesado con éxito')
                self._print('')
            
            self._print('Guardando unificado de features')
            self.df_features.to_csv(f'data/MODELS/{self.model_version}/FEATURES/df_features.csv', index = False)
        
        self._print('Proceso para obtener features finalizado con éxito')
        self._print('---------------------------------------------------------')

    def acondicionar_dataset(self):
        
        """
            Merge del dataset de objetivos con el dataset de features
        """
        self._print('Iniciando proceso para acondicionamiento')

        # Loop principal para obtener los tiempos y target en pandas
        df_ranges = pd.DataFrame()
        
        # Loop para pasar el diccionario de objetivos a formato dataframe
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

        # Merge del dataset de objetivos y de features
        self.df_final = pd.merge(self.df_features, df_ranges, how = 'inner', left_on = ['Indice','Audio_Name'], right_on = ['Indice','Audio_Name'])

        self._print('Proceso para acondicionamiento finalizado')
        self._print('---------------------------------------------------------')

    def remover_duplicados(self):
        
        self._print('Iniciando proceso para remover duplicados')

        self._print(f'Antes {len(self.df_final)}')
        self.df_final['Duplicated'] = self.df_final['Indice'].astype(str) + self.df_final['Audio_Name']
        self.df_final = self.df_final.drop_duplicates(subset = 'Duplicated')
        self.df_final = self.df_final.drop('Duplicated', axis = 1)
        self._print(f'Despues {len(self.df_final)}')

        self._print('Proceso para remover duplicados finalizado')
        self._print('---------------------------------------------------------')

    def crear_target_categorico(self):
        
        self._print('Iniciando proceso para taget categorico')

        self.df_final['Target'] = [obtener_emocion(i[0],i[1],i[2], mapping = self.mapping) for i in self.df_final['Target']]

        self._print('Proceso para target categórico finalizado')
        self._print('---------------------------------------------------------')

    def alinear_muestras(self):

        """
            Alinea las muestras a la misma cantidad, tomando como cantidad a alinear la míniea superior al threshold
        """

        self._print('Iniciando proceso para alinear muestras')

        df_copy = self.df_final.copy()
        df_copy['Cuenta'] = 1
        df_cuenta = df_copy.groupby('Target').count().reset_index()[['Target','Cuenta']]
        target_validos = df_cuenta[df_cuenta['Cuenta'] > self.min_muestras]

        threshold = target_validos['Cuenta'].min()
        
        df_final_2 = pd.DataFrame()
        for target in target_validos['Target']:
            df_temp = self.df_final[self.df_final['Target'] == target].sample(threshold)
            self._print(f'Feature: {target}, cantidad de samples: {len(df_temp)}')
            df_final_2 = pd.concat([df_temp, df_final_2])

        self.df_final = df_final_2

        self._print('Proceso para alinear muestras finalizado')
        self._print('---------------------------------------------------------')

    def entrenar(self):
        X = [i for i in self.df_final['Features'].values]
        Y = self.df_final['Target'].values

        # Encoder de las emociones
        self.encoder = OneHotEncoder()
        Y = self.encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

        # split de la data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, random_state = 42, shuffle=True)
        
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        
        self.x_train = np.expand_dims(self.x_train, axis=2)
        self.x_test = np.expand_dims(self.x_test, axis=2)

        self.model = MyKerasModel(self.x_train.shape[1], len(self.df_final['Target'].unique()), encoder = self.encoder)
        self.model.compile_model()
        self.model.train_model(self.x_train, self.y_train, self.x_test, self.y_test)
    
    def metricas_modelo(self):

        self._print('Metricas del modelo')

        loss, acc = self.model.evaluate_model(self.x_test, self.y_test)
        self._print(f'Loss: {loss}, Acc: {acc}')

        precision, recall, fscore, _ = self.model.score_model(self.x_test, self.y_test, average='macro')
        self._print('Average   : Macro')
        self._print('Precision : {}'.format(precision))
        self._print('Recall    : {}'.format(recall))
        self._print('F-score   : {}'.format(fscore))
        
        self._print('')
        
        precision, recall, fscore, _ = self.model.score_model(self.x_test, self.y_test, average='weighted')
        self._print('Average   : Weighted')
        self._print('Precision : {}'.format(precision))
        self._print('Recall    : {}'.format(recall))
        self._print('F-score   : {}'.format(fscore))

    def guardar_modelo(self):
        
        np.save(f'data/MODELS/{self.model_version}/x_train.npy', self.x_train)
        np.save(f'data/MODELS/{self.model_version}/x_test.npy', self.x_test)
        np.save(f'data/MODELS/{self.model_version}/y_train.npy', self.y_train)
        np.save(f'data/MODELS/{self.model_version}/y_test.npy', self.y_test)

        self.model.guardar_modelo(f'data/MODELS/{self.model_version}')
        
        if self.scaler is not None: 
            joblib.dump(self.scaler, f'data/MODELS/{self.model_version}/std_scaler.bin', compress=True)

        if self.encoder is not None: 
            with open(f"data/MODELS/{self.model_version}/encoder", "wb") as f: pickle.dump(self.encoder, f)

    def _convertir_strarray_a_array(self, filas):
        
        x = []
        for fila in filas:
            fila = fila.replace('[','').replace(']','').split(',')
            x.append([float(i) for i in fila])
        
        return x

    def run_pipeline(self, steps):
        for step in steps: step()