import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from sklearn.metrics import classification_report, precision_recall_fscore_support

class MyKerasModel:
    def __init__(self, input_shape, num_classes, encoder):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.encoder = encoder

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(self.input_shape, 1)))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=self.num_classes, activation='softmax'))

        return model

    def compile_model(self):
        self.model.compile(optimizer = 'adam', 
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

    def train_model(self, x_train, y_train, x_test, y_test, epochs=1, batch_size=64):
        rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, verbose=0, patience = 100, min_lr=0.0000001)
        es = EarlyStopping(monitor='val_loss', patience = 200)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[rlrp, es])

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def score_model(self, x_test, y_test, average : str = 'macro'):
        pred_test = self.model.predict(x_test)
        y_pred = self.encoder.inverse_transform(pred_test)
        y_test = self.encoder.inverse_transform(y_test)

        return precision_recall_fscore_support(y_test, y_pred, average = average)

    
    def predict(self, X):
        return self.model.predict(X)

    def guardar_modelo(self, path : str):
        self.model.save(f'{path}/model.keras')

    def guarder_reporte_estructura(self, path : str):
        # Open the file
        with open(f'{path}/report.txt', 'w', encoding="utf-8") as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

# Sacado de: https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy
class MyKerasModelv2:

    def __init__(self, input_shape, num_classes, encoder):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.encoder = encoder

    def _build_model(self):

        model = Sequential()
        model.add(Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(self.input_shape,1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5,strides=2,padding='same'))
    
        model.add(Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5,strides=2,padding='same'))
        model.add(Dropout(0.2))  # Add dropout layer after the second max pooling layer
    
        model.add(Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5,strides=2,padding='same'))
    
        model.add(Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=5,strides=2,padding='same'))
        model.add(Dropout(0.2))  # Add dropout layer after the fourth max pooling layer
    
        model.add(Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))
        model.add(Dropout(0.2))  # Add dropout layer after the fifth max pooling layer
    
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.num_classes ,activation='softmax'))

        return model

    def compile_model(self):
        self.model.compile(optimizer = 'adam', 
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])
        
    def train_model(self, x_train, y_train, x_test, y_test, epochs=1, batch_size=64):
        rlrp = ReduceLROnPlateau(monitor='val_accuracy',patience=10,verbose=1,factor=0.5,min_lr=0.00001)
        es = EarlyStopping(monitor='val_accuracy',mode='auto',patience=20,restore_best_weights=True)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[rlrp, es])

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def score_model(self, x_test, y_test, average : str = 'macro'):
        pred_test = self.model.predict(x_test)
        y_pred = self.encoder.inverse_transform(pred_test)
        y_test = self.encoder.inverse_transform(y_test)

        return precision_recall_fscore_support(y_test, y_pred, average = average)
    
    def predict(self, X):
        return self.model.predict(X)

    def guardar_modelo(self, path : str):
        self.model.save(f'{path}/model.keras')

    def guarder_reporte_estructura(self, path : str):
        # Open the file
        with open(f'{path}/report.txt', 'w', encoding="utf-8") as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))