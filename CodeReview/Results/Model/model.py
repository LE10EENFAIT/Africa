preprocessing_dir = '../Preprocessing/'
problem_dir = '../../Classes/ingestion_program/'
CRED = '\33[91m'
CEND = '\33[0m'

from sys import path; path.append(preprocessing_dir); path.append(problem_dir)
from preprocessing import Preprocessing
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from data_io import write
from data_io import zipdir
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt

class model (BaseEstimator):
    def __init__(self, classifier=RandomForestClassifier(n_estimators=200, min_samples_split=10, 
                                min_samples_leaf=2, max_features='sqrt', max_depth=None, bootstrap=True),
                        name="RandomForestClassifier"):
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.results=[] 
        self.classifier=classifier
        self.name=name
        
    def fit(self, X, y):
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        if (self.num_train_samples != num_train_samples):
            print(CRED+"ARRGH: number of samples in X and y do not match!"+CEND)
        self.classifier.fit(X, np.ravel(y))
        self.is_trained=True
        
        return self
   
    def predict(self, X):
        if X.ndim>1: num_feat = X.shape[1]
        if (self.num_feat != num_feat):
            print(CRED+"ARRGH: number of features in X does not match training data!"+CEND)
        self.results=self.classifier.predict(X)
        
        return self.results

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print(CRED+"Model reloaded from: " + modelfile+CEND)
        return self

    def getScore(self, y):
        if len(self.results) > 0:
            return roc_auc_score(y, self.results)
        else:
            print(CRED+"The model hasn't made a prediction yet."+CEND)
            return None

    def saveConfusionMatrix(self, y, filename):
        if len(self.results) > 0:
            matrice = confusion_matrix(y, self.results)
            
            fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6, 6))

            sns.heatmap(matrice, annot=True, fmt='g', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(self.name)
            ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected'])
            ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected'])
            plt.savefig(filename)
            plt.close()
        else:
            print(CRED+"The model hasn't made a prediction yet."+CEND)
        
    def exportResults(self, X_train, Y_train, X_valid, X_test, dir, filename):
        self.fit(X_train, Y_train)
        write(dir+'malaria_train.predict', self.predict(X_train))
        write(dir+'malaria_valid.predict', self.predict(X_valid))
        write(dir+'malaria_test.predict', self.predict(X_test))
        zipdir(dir+filename+".zip", dir)




def getBestClassifier(list_classifier, name_classifier, X, Y):
    Xs_train = []
    Xs_test = []
    Ys_train = []
    Ys_test = []
    for i in range(0, 20):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X, 
                                                            Y, test_size=0.33)
        Xs_train.append(X_train_pre)
        Xs_test.append(X_test_pre)
        Ys_train.append(y_train_pre)
        Ys_test.append(y_test_pre)

    scores = []
    for i in range(0, len(list_classifier)):
        scores.append(0)
    for i in range (0, len(Xs_train)):
        for j in range(0, len(list_classifier)):
            m = model(list_classifier[j], name_classifier[j])
            m.fit(Xs_train[i], Ys_train[i])
            m.predict(Xs_test[i])
            scores[j] += m.getScore(Ys_test[i])/len(Xs_train)
    max = -1
    idx = -1
    for i in range(0, len(scores)):
        if scores[i] > max:
            max = scores[i]
            idx = i
    print(scores)
    print(CRED+name_classifier[idx] + " is the best classifier of the list for these data."+CEND)
    return list_classifier[idx]
    
def getBestMetaParameters(X, Y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['sqrt', 'auto', None]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}


    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                param_distributions=random_grid, 
                                n_iter=100, 
                                cv = 3, 
                                verbose=1, 
                                random_state=None, 
                                n_jobs=-1)
    rf_random.fit(X, Y)

    for key,value in rf_random.best_params_.items():
        print(key, ':', value)

    m = model(rf_random.best_estimator_)
    m.fit(X, Y)
    return m


if __name__=="__main__":
    data = Preprocessing()
    data.compute_TSNE2D("../Preprocessing/TSNE/tsne_results2D_full.pickle")
    data.save_TSNE2D("../Images/TSNE2D_full.png")

    data.featureSelection()
    data.compute_TSNE3D("../Preprocessing/TSNE/tsne_results3D.pickle")
    data.show_TSNE3D()

    X_train_pre, X_test_pre, Y_train_pre, Y_test_pre = train_test_split(data.X_train, data.Y_train, test_size=0.33)
    m = model()
    m.load("best_random_selected")
    m.fit(X_train_pre, Y_train_pre)
    m.predict(X_test_pre)
    m.saveConfusionMatrix(Y_test_pre, "../Images/confusion_matrix_RandomForest.png")
    print(m.getScore(Y_test_pre))
    m.exportResults(data.X_train, data.Y_train, data.X_valid, data.X_test, "../Scores/Test_19_03/", "test_19_03")
    m.save("test_19_03")



