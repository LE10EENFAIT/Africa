problem_dir = '../../Classes/ingestion_program/'
data_dir = '../../malaria_data'
data_name = 'malaria'

from sys import path; path.append(problem_dir)
from data_manager import DataManager
from data_io import read_as_df
from sklearn.manifold import TSNE
import pandas as pd
from os.path import isfile
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class Preprocessing():
    def __init__(self):
        D = DataManager(data_name, data_dir, replace_missing=True)
        self.X_valid = D.data['X_valid']
        self.X_test = D.data['X_test']

        self.dataFrame = read_as_df(data_dir  + '/' + data_name)
        self.X_train = self.dataFrame.drop(['target'], axis=1)
        self.Y_train = self.dataFrame.target == 'parasitized'

    def featureSelection(self):
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(self.X_train, self.Y_train)
        feature_selection = SelectFromModel(clf, prefit=True)
        self.X_train = feature_selection.transform(self.X_train)
        self.X_test = feature_selection.transform(self.X_test)
        self.X_valid = feature_selection.transform(self.X_valid)
        self.features_idx = feature_selection.get_support(indices=True)
            
    def compute_TSNE2D(self, filename, reload=False):
        self.hue = []
        for i in self.Y_train:
            if i :
                self.hue.append('parasitized')
            else:
                self.hue.append('uninfected')

        if isfile(filename) and not reload:
            with open(filename, 'rb') as fp:
                self.tsne_results2D = pickle.load(fp)
        else:
            model = TSNE(n_components=2, verbose=True)
            self.tsne_results2D = model.fit_transform(self.X_train)
            with open(filename, 'wb') as fp:
                pickle.dump(self.tsne_results2D, fp)

        return self.tsne_results2D

    def compute_TSNE3D(self, filename, reload=False):
        self.hue3D = []
        for i in self.Y_train:
            if i :
                self.hue3D.append((0,0,1,0.5))
            else:
                self.hue3D.append((1,0.65,0,0.5))

        if isfile(filename) and not reload:
            with open(filename, 'rb') as fp:
                self.tsne_results3D = pickle.load(fp)
        else:
            model = TSNE(n_components=3, verbose=True)
            self.tsne_results3D = model.fit_transform(self.X_train)

            with open(filename, 'wb') as fp:
                pickle.dump(self.tsne_results3D, fp)

        return self.tsne_results3D
    
    def save_TSNE2D(self, filename):
        ax = plt.figure(figsize=(13,10))
        ax.suptitle("Two dimensional T-SNE algorithm on the Africa group's data")
        sns.scatterplot(x=self.tsne_results2D[:,0], y=self.tsne_results2D[:,1], hue=self.hue,legend="full",
                        alpha=0.5)
        plt.savefig(filename)
        plt.close()

    def show_TSNE3D(self):
        from mpl_toolkits.mplot3d import axes3d
        ax = plt.figure(figsize=(13,10)).gca(projection='3d')
        ax.set_title("Three dimensional T-SNE algorithm on the Africa group's data")
        ax.scatter(
            xs=self.tsne_results3D[:,0], 
            ys=self.tsne_results3D[:,1], 
            zs=self.tsne_results3D[:,2], 
            c=self.hue3D, 
            cmap='tab10'
        )
        plt.show()
        plt.close()
