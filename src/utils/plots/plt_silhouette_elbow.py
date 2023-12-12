
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config.definitions import ROOT_DIR
import matplotlib.pyplot as plt


from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np


def silhouetteplot(df):
    fig, ax = plt.subplots(3, 4, figsize=(15,8)) #plt.subplots(1, 1, figsize=(15,8))
    eixos = [(1,0), (1,1), (1,2), (1,3), 
             (2,0), (2,1), (2,2), (2,3),
             (3,0), (3,1), (3,2), (3,3)
             ] #[(1,0)]
    for c, i in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]): #[2]
        model = KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=100)
        q, mod = eixos[c]
        vis = SilhouetteVisualizer(
            model, 
            colors='yellowbrick', 
            ax=ax[q-1][mod] #comente para um plot
        )
        vis.fit(df) 
        #vis.ax.set_xlabel("Silhouette coefficient values - k = " + str(c))
        #vis.ax.legend(frameon=True, fontsize = 12)
        #vis.ax.set_title("\nk = "+ str(i)) 
        #vis.ax.set_ylabel("Cluster label")
    #vis.show()

def elbow2(X):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k, n_init=10, max_iter=100)
        kmeanModel.fit(X)
    
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
    
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                    'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_


    
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

def elbowplot(df, model, path):
    '''
    metrics = ["Distortion Score Elbow for Kmeans Clustering",
               "Silhouette Score Elbow for Kmeans Clustering",
               "Calinski Harabasz Score Elbow for Kmeans Clustering"]
    fig, ax = plt.subplots(1, 3, figsize=(15,8))
    for c, elbowmetric in enumerate(["distortion", "silhouette", "calinski_harabasz"]):
        vis = KElbowVisualizer(
            model, 
            k=(2,13), 
            metric=elbowmetric,
            locate_elbow=True, 
            timings=False,
            ax=ax[c]
            )
        vis.fit(df)    
        vis.ax.set_xlabel("K")
        vis.ax.legend(frameon=True, fontsize = 12)
        vis.ax.set_title(metrics[c]) 
        vis.ax.set_ylabel(elbowmetric + "metric")
    return vis.show()
    '''
    ftsz = 30
    fig, ax = plt.subplots(1, 1, figsize=(15,8))
    vis = KElbowVisualizer(
        model, 
        k=(2,10), 
        metric="silhouette",
        locate_elbow=True, 
        timings=False,
        colors='black'

        #ax=ax[c]
    )
    vis.fit(df)    

    if vis.elbow_value_:
        vis.ax.lines[0].set_color('black')
        vis.ax.lines[0].set_linewidth(5)
        vis.ax.lines[1].set_linewidth(5)

    vis.ax.legend(frameon=True, fontsize = 25)
    vis.ax.set_xlabel("K", fontsize=ftsz)
    #plt.title("Yelp - Silhouete score elbow for kmeans clustering", fontsize = ftsz) 
    vis.ax.set_ylabel("silhouette score", fontsize=ftsz)
    vis.fig.set_size_inches(10,6)

    for label in ax.get_yticklabels():
        label.set_size(ftsz)
    for label in ax.get_xticklabels():
        label.set_size(ftsz)
    plt.savefig(ROOT_DIR + f"/src/results/{path}/clustering_analysis/DEFC_silhouette.png", bbox_inches='tight')
    plt.show()
    return vis.show()