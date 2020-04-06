from sys import path
path.append('ingestion_program/')

from data_manager import DataManager
from data_io import write
from data_io import zipdir

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization,Dropout

class model:
    def __init__(self, size=50, type_c=1):
        self.size = size
        self.is_trained = False
        
        if type_c == 1:
            self.classifier = Sequential()
            self.classifier.add(Convolution2D(32, (3, 3), input_shape = (self.size, self.size, 1), activation = 'relu'))
            self.classifier.add(Convolution2D(64, (3, 3), activation='relu'))
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            self.classifier.add(Dropout(0.25))
            self.classifier.add(Flatten())
            self.classifier.add(Dense(units=128, activation='relu'))
            self.classifier.add(Dropout(0.5))
            self.classifier.add(Dense(activation = 'sigmoid', units=1))
            self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            
        else:
            self.classifier = Sequential()
            self.classifier.add(Convolution2D(32, (3, 3), input_shape = (self.size, self.size, 1), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
            self.classifier.add(BatchNormalization(axis = -1))
            self.classifier.add(Dropout(0.2))
            self.classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
            self.classifier.add(BatchNormalization(axis = -1))
            self.classifier.add(Dropout(0.2))
            self.classifier.add(Flatten())
            self.classifier.add(Dense(activation = 'relu', units=512))
            self.classifier.add(BatchNormalization(axis = -1))
            self.classifier.add(Dropout(0.2))
            self.classifier.add(Dense(activation = 'relu', units=256))
            self.classifier.add(BatchNormalization(axis = -1))
            self.classifier.add(Dropout(0.2))
            self.classifier.add(Dense(activation = 'sigmoid', units=1))
            self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    def transform_results(self, res):
        results = []
        append = results.append
        for pred in res:
            if pred > 0.5:
                append(1)
            else:
                append(0)
        return results

    def fit(self, X, y, epochs=50):
        X = X.reshape(len(X), self.size, self.size, 1)
        self.classifier.fit(X,
                        y,
                        batch_size = self.size,
                        epochs=epochs,
                        shuffle = False
                    )
        self.is_trained = True

    def predict(self, X):
        X = X.reshape(len(X), self.size, self.size, 1)
        if self.is_trained:
            return self.transform_results(self.classifier.predict(X))
        else:
            print("The model is not trained yet.")
            return False

    def getScore(self, X, y):
        res = self.predict(X)
        return roc_auc_score(res, y)

    
    def exportResults(self, X_train, Y_train, X_valid, X_test, dir, filename, epochs=50):
        self.fit(X_train, Y_train, epochs=epochs)
        write(dir + "malaria_train.predict", self.predict(X_train))
        write(dir + "malaria_valid.predict", self.predict(X_valid))
        write(dir + "malaria_test.predict", self.predict(X_test))
        zipdir(dir + filename + ".zip", dir)

if __name__ == "__main__":

    D = DataManager("malaria", "malaria_input_data", replace_missing=True)
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']

    X_valid = D.data['X_valid']
    X_test = D.data['X_test']

    model = model()

    model.exportResults(X_train, Y_train, X_valid, X_test, "Score/", "CNN_20_epochs", epochs=20)

