'''
    It is paid for obvious reasons by fillipe.silva
'''

import numpy as np
import json

#To log wan_ls
import ast
from collections import ChainMap

import numba
from numba import jit, cuda

import wandb
from config.definitions import WANDB_API

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
pandas2ri.activate()

import tensorflow as tf

clue = importr("clue")
bootcluster = importr("bootcluster")
OTclust = importr("OTclust")

rng = np.random.RandomState(1)

class StabilityGPU:

    # function optimized to run on gpu 
    #@jit(target_backend='cuda')
    @numba.jit(nopython=True)
    def run(self, 
            data, 
            clusters, 
            size_sample, 
            n_samples, 
            epochs,
            path_to_save,
            project_name,
            otclust,
            random_state = None,
            ):
        
        try:
            print("[Running on a GPU mode]")
            print("Num. GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        except RuntimeError as e:
            print(e, "Visible devices must be set before GPUs have been initialized")
        
        
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project = project_name,
            
        )
          
        stab_epochs = {}
        arguments = {
            "clusters": clusters, 
            "epochs": epochs,
            "n_samples": n_samples,
            "random_state": random_state
        }
        stab_methods = {
            "adjusted_rand_score": [],
            "adjusted_mutual_info_score": [],
            "bagclust": [],
            "han": [],
            "OTclust": [],
        }

        rData = None
        with localconverter(ro.default_converter + pandas2ri.converter):
            rData = ro.conversion.rpy2py(data)

        with tf.device('/device:GPU:0'):
            for ep in range(epochs):
                print("[>] ep:", ep + 1)
                for cluster in clusters:
                    print("[>] ep:", ep + 1, "cluster: ", cluster)

                    kmeans = KMeans(n_clusters=cluster, n_init=10)

                    labels, indices = self.get_labels_and_indices(data, kmeans, size_sample, n_samples, random_state)        
                    stab_methods['adjusted_rand_score'].append(self.adjusted_rand_score(labels, indices, cluster))
                    stab_methods['adjusted_mutual_info_score'].append(self.adjusted_mutual_info_score(labels, indices, cluster))
                    stab_methods['bagclust'].append(self.bagclust(rData, n_samples, cluster))
                    stab_methods['han'].append(self.han(rData, n_samples, cluster))
                    if otclust: stab_methods['OTclust'].append(self.OTstab(rData, n_samples, cluster))
                    
                wan_ls = []
                for k,v in enumerate(clusters):
                    wan_ls.append(ast.literal_eval(f"{{'adjusted_rand_score: k({v})': {stab_methods['adjusted_rand_score'][k]}}}"))
                    wan_ls.append(ast.literal_eval(f"{{'adjusted_mutual_info_score: k({v})': {stab_methods['adjusted_mutual_info_score'][k]}}}"))
                    wan_ls.append(ast.literal_eval(f"{{'bagclust: k({v})': {stab_methods['bagclust'][k]}}}"))
                    wan_ls.append(ast.literal_eval(f"{{'han: k({v})': {stab_methods['han'][k]}}}"))
                    if otclust: wan_ls.append(ast.literal_eval(f"{{'OTclust: k({v})': {stab_methods['OTclust'][k]}}}"))
                
                wandb.log(dict(ChainMap(*wan_ls)))

                stab_epochs[ep] = stab_methods
                stab_methods = {"adjusted_rand_score": [],"adjusted_mutual_info_score": [],"bagclust": [],"han": [],"OTclust": []}
                    
            self.save(stab_epochs, arguments, path_to_save)

    def get_labels_and_indices(self, data, clrt_algorithm, size_sample, n_samples, random_state):
        labels = []
        indices = []
        for _ in range(n_samples):
            # draw bootstrap samples, store indices
            sample_indices = rng.randint(0, data.shape[0], size_sample)
            indices.append(sample_indices)
            clrt_algorithm = clone(clrt_algorithm)
            if hasattr(clrt_algorithm, "random_state"):
                # randomize estimator if possible
                clrt_algorithm.random_state = rng.randint(1e5)
            data_bootstrap = data[sample_indices]
            clrt_algorithm.fit(data_bootstrap)
            # store clustering outcome using original indices
            relabel = -np.ones(data.shape[0], dtype=int)
            relabel[sample_indices] = clrt_algorithm.labels_
            labels.append(relabel)
        return (labels, indices)

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    def adjusted_rand_score(self, labels, indices, cluster):
        print("Computing [adjusted_rand_score]")
        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                in_both = np.intersect1d(i, j)
                scores.append(adjusted_rand_score(l[in_both], k[in_both])) 
        return np.mean(scores)

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    def adjusted_mutual_info_score(self, labels, indices, cluster):
        print("Computing [adjusted_mutual_info_score]")
        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                in_both = np.intersect1d(i, j)
                scores.append(adjusted_mutual_info_score(l[in_both], k[in_both]))
        return np.mean(scores)

    #"A prediction-based resampling method for estimating the number of clusters in a dataset."
    #"Bagging to improve the accuracy of a clustering procedure."
    #Explicação: Stability estimation for unsupervised clustering: A review
    def bagclust(self, rData, n_samples, n_cluster):
        print("Computing [bagclust]")
        rDataStab = clue.cl_bag(x = rData, B = n_samples, k = n_cluster)
        return rDataStab.rx2['.Data'].max(axis = 1).mean()

    #Bootstrapping estimates of stability for clusters,observations and model selection
    #Para entender vá para a página 4 Seção 2 Fig. 1.
    def han(self, rData, n_samples, n_cluster):
        print("Computing [han]")
        try:
            hanStab = bootcluster.stability(x = rData, k = n_cluster, B = n_samples)
            return hanStab.rx2['overall']
        except RRuntimeError:
            return float(0)

    #CPS Analysis for cluster validation
    #Install from github https://github.com/cran/OTclust
    #Melhor explicação: Denoising Methods for Inferring Microbiome Community Content and Abundance
    def OTstab(self, rData, n_samples, n_cluster): 
        print("Computing [OTstab]")
        otclust = OTclust.clustCPS(rData, k=n_cluster, l= False, pre=False, noi="after",
                 nPCA = 2, nEXP = n_samples)
        return float(otclust.rx2['tight_all'])

    def save(self, stab_epochs, arguments, path_to_save):
        data = json.dumps(stab_epochs, indent = 4)
        file = open(path_to_save,"w")
        file.write(data)
        file.close()