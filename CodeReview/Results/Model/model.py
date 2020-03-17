score_dir = '../../Classes/scoring_program/'
preprocessing_dir = '../Preprocessing/'
CRED = '\33[91m'
CEND = '\33[0m'

from sys import path; path.append(score_dir); path.append(preprocessing_dir);
from preprocessing import Preprocessing
from libscores import get_metric
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
metric_name, scoring_function = get_metric()

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
        num_test_samples = X.shape[0]
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
            return scoring_function(y, self.results)
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


def getBestClassifier(list_classifier, name_classifier, data):
    from sklearn.model_selection import train_test_split
    Xs_train = []
    Xs_test = []
    Ys_train = []
    Ys_test = []
    for i in range(0, 20):
        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(data.X_train, 
                                                            data.Y_train, test_size=0.33)
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
    print(CRED+name_classifier[idx] + " is the best classifier of the list for these data."+CEND)
    return list_classifier[idx]
    
    


if __name__=="__main__":
    data = Preprocessing()
    data.featureSelection()
    data.compute_TSNE3D("../Preprocessing/TSNE/tsne_results3D.pickle")
    data.compute_TSNE2D("../Preprocessing/TSNE/tsne_results2D.pickle")
    data.show_TSNE3D()
    data.save_TSNE2D("../Images/TSNE2D.png")

    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    temp = getBestClassifier([KNeighborsClassifier(), MLPClassifier(solver='lbfgs'), SGDClassifier()],
                            ['KNeighbors', "MLP", "SGDC"], data)
    

