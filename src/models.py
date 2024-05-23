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
        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(self.input_shape, 1)))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=self.num_classes, activation='softmax'))

        return model

    def compile_model(self):
        self.model.compile(optimizer = 'adam', 
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

    def train_model(self, x_train, y_train, x_test, y_test, epochs=1, batch_size=64):
        rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        es = EarlyStopping(monitor='val_loss', patience=5)
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