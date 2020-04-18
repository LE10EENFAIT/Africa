'''
Challenge MEDICHAL : Team AFRICA

Serre Gaëtan
Patault Paul
Estevan Benjamin
Ferreira Jules
Ndao Ndiémé
Iskounen Feriel

Dernier changement : 18/04/2020

Ajouts : Des tests, predict_proba, save_roc_curve

'''



from sys import path

path.append("Classes/ingestion_program/")
from preprocessing import Preprocessing
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from data_io import write
from data_io import zipdir
from data_manager import DataManager
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt


def print_red(stri):
    """
    Fonction qui permet un affichage coloré en ROUGE.
    On l'utilise pour différencier nos différents print dans la console.
    """
    CRED = "\33[91m"
    CEND = "\33[0m"
    print(CRED + stri + CEND)


class model:
    """
    Classe de la partie Modèle.
    Elle implémente les principales méthodes liées aux classifieurs tels que fit, predict, predict_proba, etc.
    """

    def __init__(
        self,
        """
        Le modèle chargé est très simple, mais c'est celui avec lequel nous avons obtenu nos meilleurs scores
        """
        classifier=RandomForestClassifier(
            n_estimators=600
        ),
        name="RandomForestClassifier",
        load=""
    ):
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        self.is_trained = False
        self.results = []
        self.classifier = classifier
        self.name = name
        self.preprocessor = Preprocessing()
        if load != "":
            modelfile = load + "_model.pickle"
            if isfile(modelfile):
                with open(modelfile, "rb") as f:
                    self = pickle.load(f)

    def fit(self, X, y, outliers=False):
        """
        Entraine le modèle chargé dans self.classifier sur les données (X, y).
        """
        if outliers:
            X, y = self.preprocessor.outliers_filtering(X, y)
        else:
            self.preprocessor.fit(X, y)
        X = self.preprocessor.fit_transform(X)
        self.num_train_samples = X.shape[0]
        if X.ndim > 1:
            self.num_feat = X.shape[1]
        num_train_samples = y.shape[0]
        if y.ndim > 1:
            self.num_labels = y.shape[1]
        if self.num_train_samples != num_train_samples:
            print_red("ARRGH: number of samples in X and y do not match!")
        self.classifier.fit(X, np.ravel(y))
        self.is_trained = True

        return self

    def predict(self, X):
        """
        Prédiction des données X avec le modèle self.classifier.
        Return la liste des labels prédits.
        """
        X = self.preprocessor.fit_transform(X)
        if self.is_trained:
            if X.ndim > 1:
                num_feat = X.shape[1]
            if self.num_feat != num_feat:
                print_red(
                    "ARRGH: number of features in X does not match training data!"
                )
            self.results = self.classifier.predict(X)

            return self.results
        else:
            print_red("The model is not trained yet")
            return None
            
    
    def predict_proba(self, X):
        """
        Prédiction des données X avec le modèle self.classifier.
        Return la liste des labels prédits.
        """
        X = self.preprocessor.fit_transform(X)
        if self.is_trained:
            if X.ndim > 1:
                num_feat = X.shape[1]
            if self.num_feat != num_feat:
                print_red(
                    "ARRGH: number of features in X does not match training data!"
                )

            self.results = self.classifier.predict_proba(X)

            return self.results
        else:
            print_red("The model is not trained yet")
            return None
        

    def save(self, path="./"):
        """
        On enregistre le modèle dans un fichier .pickle afin de ne pas devoir le réentrainer à chaque éxecution.
        """
        pickle.dump(self, open(path + "_model.pickle", "wb"))

    def load(self, path="./"):
        """
        Possibilité de charger un modèle (pré-entrainé) dans self.classifier
        """
        modelfile = path + "_model.pickle"
        if isfile(modelfile):
            with open(modelfile, "rb") as f:
                self = pickle.load(f)
            print_red("Model reloaded from: " + modelfile)
        return self

    def getScore(self, y):
        """
        Calcul du score du modèle à partir de self.results et de la fontion roc_curve.
        Return le score sur [0, 1] si le modèle a été entrainé, None sinon.
        """
        if len(self.results) > 0:
            fpr, tpr, thresholds = roc_curve(y, self.results[:, 1])
            roc_auc = auc(fpr, tpr)
            return roc_auc
        else:
            print_red("The model hasn't made a prediction yet.")
            return None

    def saveConfusionMatrix(self, y, filename):
        """
        Enregistrement la matrice de confusion.
        Return True si la sauvegarde a été faite, False sinon
        """
        if len(self.results) > 0:
            matrice = confusion_matrix(y, self.results)

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

            sns.heatmap(matrice, annot=True, fmt="g", ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title(self.name)
            ax.xaxis.set_ticklabels(["Parasitized", "Uninfected"])
            ax.yaxis.set_ticklabels(["Parasitized", "Uninfected"])
            plt.savefig(filename)
            plt.close()
            print_red("Confusion matrix saved in : " + filename)
            return True
        else:
            print_red("The model hasn't made a prediction yet.")
            return False
            
    def save_roc_curve(self, y, filename):
        """
        Enregistrement la ROC curve.
        Return True si l'image a été sauvegardée, False sinon
        """
        if len(self.results) > 0:
            fpr, tpr, thresholds = roc_curve(y, self.results[:, 1])
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
            print_red("ROC curve saved in : " + filename)
            
            return True
        else:
            print_red("The model hasn't made a prediction yet.")
            return False

    def exportResults(self, X_train, X_valid, X_test, dir, filename):
        """
        Exporte les prédictions calculées sur (X_train, X_valid, X_test) dans un dossier zippé.
        Return True si les résultats ont été exportés, False sinon
        """
        if self.is_trained:
            write(dir + "malaria_train.predict", self.predict(X_train))
            write(dir + "malaria_valid.predict", self.predict(X_valid))
            write(dir + "malaria_test.predict", self.predict(X_test))
            zipdir(dir + filename + ".zip", dir)
            return True
        else:
            print("The model is not trained yet.")
            return False

def getBestClassifier(list_classifier, name_classifier, X, Y):
    """
    Calcul les scores de chaque classifieur de 'list_classifier' sur différents ensembles de données issus de (X, Y).
    On renvoie alors le classifieur qui a le meilleur score moyen sur ces jeux. 
    """

    # tableaux de données
    Xs_train = []
    Xs_test = []
    Ys_train = []
    Ys_test = []

    # on découpe les données de 20 façons différentes
    # qu'on charge dans nos tableaux
    for i in range(0, 20):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            X, Y, test_size=0.33
        )
        Xs_train.append(X_train_pre)
        Xs_test.append(X_test_pre)
        Ys_train.append(y_train_pre)
        Ys_test.append(y_test_pre)

    # on calcul le score pour chaque classifieur sur chacun des jeux de données
    # et on en fait la moyenne
    scores = []
    for i in range(0, len(list_classifier)):
        scores.append(0)
    for i in range(0, len(Xs_train)):
        for j in range(0, len(list_classifier)):
            m = model(list_classifier[j], name_classifier[j])
            m.fit(Xs_train[i], Ys_train[i])
            m.predict_proba(Xs_test[i])
            scores[j] += m.getScore(Ys_test[i]) / len(Xs_train)

    # on choisit le meilleur classifieur à partir des scores moyens obtenus
    max = -1
    idx = -1
    for i in range(0, len(scores)):
        if scores[i] > max:
            max = scores[i]
            idx = i
    print_red(
        name_classifier[idx] + " is the best classifier of the list for these data."
    )
    return list_classifier[idx]


def getBestMetaParameters(X, Y):
    """
    On cherche les meilleurs méta-paramètre d'une RandomForest avec la fonction RandomizedSearchCV (de sklearn) sur les données (X, Y).
    Renvoie le modèle entrainé ainsi obtenu.
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]

    # Number of features to consider at every split
    max_features = ["sqrt", "auto", None]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=None,
        n_jobs=-1,
    )
    rf_random.fit(X, Y)

    for key, value in rf_random.best_params_.items():
        print(key, ":", value)

    m = model(rf_random.best_estimator_)
    m.fit(X, Y)
    return m
    
if __name__ == "__main__":

    #Batterie de tests

    D = DataManager("malaria", "malaria_data", replace_missing=True)
    
    # Chargement des données brutes
    data = Preprocessing()
    data.X_train = D.data['X_train']
    data.Y_train = D.data['Y_train']
    data.X_valid = D.data['X_valid']
    data.X_test = D.data['X_test']
    

    #On test si nos labels sont bien seulement parasitized (1) et uninfected (0)
    from sklearn.utils.multiclass import unique_labels
    import collections
    import sys
    if collections.Counter(unique_labels(data.Y_train)) != collections.Counter([0,1]):
        print("There is an error in labels")
        sys.exit()



    X_train_pre, X_test_pre, Y_train_pre, Y_test_pre = train_test_split(
        data.X_train, data.Y_train, test_size=0.33
    )


    # Chargement d'un modèle calculé au préalable AVEC preprocessing
    m = model()
    m.fit(X_train_pre, Y_train_pre)
    m.predict_proba(X_test_pre)
    #m.saveConfusionMatrix(Y_test_pre, "Results/Images/confusion_matrix_RandomForest_selected.png")
    m.save_roc_curve(Y_test_pre, "Results/Images/roc_curve_RandomForest_selected.png")
    
    print_red("Score for best_random_selected_model is " + str(m.getScore(Y_test_pre)))

    m.exportResults(
        data.X_train,
        data.X_valid,
        data.X_test,
        "Results/Scores/Best_Random/",
        "best_random_selected",
    )

    data.fit(data.X_train, data.Y_train)
    data.X_train = data.fit_transform(data.X_train)
    data.compute_TSNE2D("Results/Preprocessing/TSNE/tsne_results2D.pickle")
    data.save_TSNE2D("Results/Images/TSNE2D.png")

    data.saveDecisionSurface("Results/Images/DecisionSurface.png")
    data.compute_TSNE3D("Results/Preprocessing/TSNE/tsne_results3D.pickle")
    data.show_TSNE3D()

    # On regarde quel est le meilleur classifieur parmi (RandomForest, MLP, KNeighbors) et on affiche son score
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB

    m = model(
        getBestClassifier(
            [RandomForestClassifier(), MLPClassifier(solver="lbfgs"), GaussianNB()],
            ["RandomForest", "MLP", "KNeighbors"],
            data.X_train,
            data.Y_train,
        )
    )
    m.fit(X_train_pre, Y_train_pre)
    m.predict_proba(X_test_pre)
    print_red(
        "Score for the best classifier of the list is " + str(m.getScore(Y_test_pre))
    )
    m.save("Results/Model/test")
