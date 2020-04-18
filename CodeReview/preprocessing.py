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

from sklearn.manifold import TSNE
import pandas as pd
from os.path import isfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest

#Les tests sont dans le main du fichier model.py


class Preprocessing():
   
    # _init_:
    #      self: Instance of the Preprocessing class 
    #   
    #   We initialise a new instance and give it the four attributes
    #   X_valid, X_test, X_train and Y_train as arrays containing 
    #   the corresponding data from the malaria dataset.
    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.X_valid = []
        self.X_test = []
    
    # fit & fit_transform:
    #      self: Instance of the Preprocessing class 
    #
    #   By going through a randomForest of 50 trees,
    #   we estimate which features have the most importance
    #   in the calculation of the score and we remove from
    #   the arrays the columns of the features that have the least
    #   influence on the calculation of the score.
    
    def fit(self, X_train, Y_train):
        self.clf = ExtraTreesClassifier(n_estimators=50)
        self.clf = self.clf.fit(X_train, Y_train)

    def fit_transform(self, X):
        feature_selection = SelectFromModel(self.clf, prefit=True)
        return feature_selection.transform(X)
    
    # outliers_filtering:
    #      self: Instance of the Preprocessing class
    #      X_train: Array containing the training data
    #      Y_train: Array containing the training label
    #   
    #   We analyse the training data and remove most outliers from it
    #   thanks to the IsolationForest model that let us detect them
    def outliers_filtering(self, X_train, Y_train):
        rows, cols = X_train.shape
        nb_0 = np.sum(Y_train[:] == Y_train[0])
        nb_1 = np.sum(Y_train[:] != Y_train[0])
        
        class_0 = np.zeros((nb_0, cols))
        idx_0 = 0
        class_1 = np.zeros((nb_1, cols))
        idx_1 = 0
        for i in range(rows):
            if(Y_train[i] == Y_train[0]):
                for j in range(cols):
                    class_0[idx_0][j] = X_train[i][j]
                idx_0 = idx_0+1
            if(Y_train[i] != Y_train[0]):
                for j in range(cols):
                    class_1[idx_1][j] = X_train[i][j]
                idx_1 = idx_1+1
        
        clf_0 = IsolationForest(n_estimators=10, warm_start=True)
        C0 = clf_0.fit_predict(class_0, y=None)
        nb_inliers = 0
        for i in range(nb_0):
            if(C0[i] == 1):
                nb_inliers = nb_inliers+1
        nb_0 = nb_inliers
        clf_1 = IsolationForest(n_estimators=10, warm_start=True)
        C1 = clf_1.fit_predict(class_1,y=None)
        for i in range(nb_1):
            if(C1[i] == 1):
                nb_inliers = nb_inliers+1
        nb_1 = nb_inliers-nb_0
        
        # After the Random forests have detected all the inliers, 
        # we put them in a new array to create a new dataset without outliers.
        X = np.zeros((nb_inliers, cols+1))
        idx = 0
        for i in range(C0.size):
            if(C0[i] == 1):
                for j in range(cols):
                    X[idx][j] = class_0[i][j]
                idx = idx+1
        for i in range(C1.size):
            if(C1[i] == 1):
                for j in range(cols):
                    X[idx][j] = class_1[i][j]
                X[idx][cols] = 1
                idx = idx+1
                
        # Finally we shuffle the lines of the dataset
        np.random.shuffle(X)
        X_train = np.zeros((nb_inliers, cols))
        for i in range(nb_inliers):
            for j in range(cols):
                X_train[i][j] = X[i][j]
        Y_train = np.zeros(nb_inliers)
        for i in range(nb_inliers):
            Y_train[i] = X[i][cols]
        self.fit(X_train, Y_train)
        return (X_train,Y_train)

            
    # compute_TSNE2D:
    #       self: Instance of the Preprocessing class 
    #       filename: String containing the name of a pickle
    #       reload: boolean to tell the function to load or not an existing pickle
    #
    #   Attributes a list hue and an array tsne_results2D to the instance of Preprocessing.
    #   Fills hue with lists containing the color of each data depending on its label.
    #   If there is no pickle or if reaload=True, trains a model to give 3D coordinates
    #   to each of our data and put them in the tsne_results3D array before dumping it into
    #   a new pickle named as filename and returning it.
    #   Else, reads the array from the pickle named as filename and returns it.    
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
    
    # compute_TSNE3D:
    #       self: Instance of the Preprocessing class 
    #       filename: String containing the name of a pickle
    #       reload: boolean to tell the function to load or not an existing pickle
    #
    #   Attributes a list hue3D and an array tsne_results3D to the instance of Preprocessing.
    #   Fills hue3D with lists containing the color of each data depending on its label.
    #   If there is no pickle or if reaload=True, trains a model to give 3D coordinates
    #   to each of our data and put them in the tsne_results3D array before dumping it into
    #   a new pickle named as filename and returning it.
    #   Else, reads the array from the pickle named as filename and returns it.
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
    
    # save_TSNE2D:
    #       self: Instance of the Preprocessing class 
    #       filename: String containing the name of a file
    #   
    #   Using the tsne_results2D(coordinates) and the hue(colors) attributes 
    #   of the instance, draws a 2D graph of the T-SNE of the dataset, calls it
    #   "Two dimensional T-SNE algorithm on the Africa group's data" and saves 
    #   it into a file named as filename.
    def save_TSNE2D(self, filename):
        ax = plt.figure(figsize=(13,10))
        ax.suptitle("Two dimensional T-SNE algorithm on the Africa group's data")
        sns.scatterplot(x=self.tsne_results2D[:,0], y=self.tsne_results2D[:,1], hue=self.hue,legend="full",
                        alpha=0.5)
        plt.savefig(filename)
        plt.close()
        print("T-SNE2D saved in " + filename)

        
    # show_TSNE3D:
    #       self: Instance of the Preprocessing class
    #   
    #   Using the tsne_results3D(coordinates) and the hue3D(colors) attributes 
    #   of the instance, draws a 3D graph of the T-SNE of the dataset, calls it
    #   "Three dimensional T-SNE algorithm on the Africa group's data" and
    #   displays it.
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
          

     # saveDecisionSurface:
     # Plot and save the decision surface of a decision tree trained on pairs of features 
     #  self: Instance of the Preprocessing class  
     #  filename: String containing the name of a file 

    def saveDecisionSurface(self, filename):
        # Parameters
        n_classes = 2
        plot_colors = ['g','b']
        plot_edgecolors = ['r','y']
        plot_markers = ['s','o']
        plot_classes = ['parasitized', 'uninfected']
        plot_step = 0.02
        plt.figure(figsize=(15,10))
        liste = []
        for i in range(0, len(self.X_train[0])-1):
            for j in range(i+1, len(self.X_train[0])):
                temp = [i]
                temp.append(j)
                liste.append(temp)

        for pairidx, pair in enumerate(liste):
            # We only take the two corresponding features
            X = self.X_train[:,pair] 
            y = self.Y_train
            

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std
            
            # Train
            clf = DecisionTreeClassifier().fit(X, y)

            # Plot the decision boundary
            plt.subplot(2, 3, pairidx + 1)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                np.arange(y_min, y_max, plot_step))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

            plt.axis("tight")

            # Plot the training points
            for i, color, edge_color, marker, classe in zip(range(n_classes), plot_colors, plot_edgecolors, plot_markers, plot_classes):
                idx = np.where(y == i)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, edgecolor=edge_color, marker=marker, label=classe,
                            cmap=plt.cm.Paired)

            plt.axis("tight")

        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        print("Surface decisionTree saved in " + filename)

