'''
Challenge MEDICHAL : Team AFRICA

Serre Gaëtan
Patault Paul
Estevan Benjamin
Ferreira Jules
Ndao Ndiémé
Iskounen Feriel

Dernier changement : 18/04/2020

'''

from sys import path
path.append('ingestion_program/')

from data_manager import DataManager
from data_io import write
from data_io import zipdir

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization, Dropout, AveragePooling2D

import matplotlib.pyplot as plt

class model:
    """
    Classe de la partie Modèle.
    Elle implémente les principales méthodes liées aux classifieurs tels que fit, predict_proba, etc.
    Deux CNN sont disponibles
    """

    
    def __init__(self, size=50, type_c=1):
        self.size = size
        self.is_trained = False
        self.history = None
        self.results = []
        
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
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.2))
            self.classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
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
    
    def summary(self):
        """
        Affiche un résumé du CNN de notre modèle
        """
        self.classifier.summary()
    
    def save_model_image(self, filename):
        """
        Sauvegarde un schéma du CNN de notre modèle
        """
        keras.utils.plot_model(self.classifier, show_layer_names=False, to_file=filename)
    
    def save_model(self, filename):
        """
        Sauvegarde notre CNN et ses poids dans filename
        """
        self.classifier.save(filename)
    
    def load_model(self, filename):
        """
        Charge un CNN et ses poids à partir de filename
        """
        self.classifier = load_model(filename)
        self.is_trained = True
    
    def save_history(self, filename):
        """
        Sauvegarde une image correspondant à l'historique du score sur l'ensemble d'entrainement pendant fit
        """
        if self.history != None:
            plt.plot(self.history.history['accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig(filename)
            plt.close()
            #plt.show()
        else:
            print("The model is not trained yet.")
            return False
    
    

    def fit(self, X, y, epochs=50):
        """
        Entraine le modèle chargé dans self.classifier sur les données (X, y).
        """
        X = X.reshape(len(X), self.size, self.size, 1)
        self.history = self.classifier.fit(X, y, epochs=epochs)
        self.is_trained = True

    def predict_proba(self, X):
        """
        Prédiction des données X avec le modèle self.classifier.
        Return la liste des labels prédits.
        """
        X = X.reshape(len(X), self.size, self.size, 1)
        if self.is_trained:
            self.results = self.classifier.predict(X)
            return self.results
        else:
            print("The model is not trained yet.")
            return False

    def getScore(self, X, y):
        """
        Calcul du score du modèle à sur X y et de la fontion roc_auc_score.
        """
        res = self.predict_proba(X)
        
        fpr, tpr, thresholds = roc_curve(y, res)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    
    
    def exportResults(self, X_train, X_valid, X_test, dir, filename):
        """
        Exporte les prédictions calculées sur (X_train, X_valid, X_test) dans un dossier zippé.
        Return True si les résultats ont été exportés, False sinon
        """
        if self.is_trained:
            write(dir + "malaria_train.predict", self.predict_proba(X_train))
            write(dir + "malaria_valid.predict", self.predict_proba(X_valid))
            write(dir + "malaria_test.predict", self.predict_proba(X_test))
            zipdir(dir + filename + ".zip", dir)
            return True
        else:
            print("The model is not trained yet.")
            return False
        
    def save_roc_curve(self, y, filename):
        """
        Enregistrement la ROC curve.
        Return True si l'image a été sauvegardée, False sinon
        """
        if len(self.results) > 0:
            fpr, tpr, thresholds = roc_curve(y, self.results)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 7))

            plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.savefig(filename)
            plt.close()
            print("ROC curve saved in : " + filename)
            
            return True
        else:
            print_red("The model hasn't made a prediction yet.")
            return False

if __name__ == "__main__":
    
    """
    Batterie de tests
    """

    D = DataManager("malaria", "malaria_input_data", replace_missing=True)
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']

    X_valid = D.data['X_valid']
    X_test = D.data['X_test']
    
    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_train, Y_train, test_size=0.33)

    #Création d'un modèle. Entraînement sur 5 époques puis prédiction et affichage du score.
    m = model()
    m.summary()
    m.fit(X_train_pre, y_train_pre, epochs=5)
    m.save_history('Images/histo.png')
    print(m.getScore(X_test_pre, y_test_pre))
    m.save_roc_curve(y_test_pre, 'Images/roc_curve_5_epoch.png')

    #Chargement d'un modèle entraîné au préalable sur 50 époques avec une NVidia GTX 1080. Prédiction, affichage du score puis exportation des résultats vers un format pour codalab
    m = model()
    m.load_model('Models/model1_50_epochs.h5')
    m.summary()
    print(m.getScore(X_test_pre, y_test_pre))
    m.save_roc_curve(y_test_pre, 'Images/roc_curve_100_epochs.png')
    m.exportResults(X_train, X_valid, X_test, "Scores/", "CNN_50_epochs")
