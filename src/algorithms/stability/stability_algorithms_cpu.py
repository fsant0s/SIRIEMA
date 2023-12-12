import os
import numpy as np
import json
import logging
import math

import ast
from collections import ChainMap

import wandb
from config.definitions import WANDB_API

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from multiprocessing import Process, Manager

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
pandas2ri.activate()

clue = importr("clue")
bootcluster = importr("bootcluster")
OTclust = importr("OTclust")

rng = np.random.RandomState(1)
logger = logging.getLogger(__name__)

class StabilityCPU:

    """
    Class to compute stability using CPU
    """

    def __init__(self, stability_args, ROOT_DIR):
        self.stability_args = stability_args
        self.ROOT_DIR = ROOT_DIR

    def run(self, 
            data, 
    ):
        """
            param: data
            size_sample: size_sample
        """
        data = data 
        size_sample = data.shape[0]
        clusters = self.stability_args.clusters
        num_bootstrap_samples = self.stability_args.num_bootstrap_samples
        num_train_epochs = self.stability_args.num_train_epochs
        path_and_file_name_to_save = self.ROOT_DIR + self.stability_args.output_dir +  self.stability_args.output_stab_name
        project_name = self.stability_args.project_name
        random_state = self.stability_args.RNDN

        logger.info("[Running on a CPU mode]")
        
        # start a new wandb run to track this script
        if self.stability_args.report_to:
            wandb.init(
                project = project_name,
            )

        manager = Manager()
        stab_methods = manager.dict()
        stab_methods["adjusted_rand_score"] = manager.list()
        stab_methods["adjusted_mutual_info_score"] = manager.list()
        stab_methods["bagclust"] = manager.list()
        stab_methods["han"] = manager.list()
        stab_methods["OTclust"] = manager.list()

        arguments = {
            "clusters": clusters, 
            "eponum_train_epochschs": num_train_epochs,
            "num_bootstrap_samples": num_bootstrap_samples,
            "random_state": random_state
        }
        stab_epochs = {}

        rData = None
        with localconverter(ro.default_converter + pandas2ri.converter):
            rData = ro.conversion.rpy2py(data)
    
        for ep in range(num_train_epochs):
            logger.info(f"[>] ep: {ep + 1}")
            for cluster in clusters:
                logger.info(f"[>] ep: {ep + 1} cluster: {cluster}")

                kmeans = KMeans(n_clusters=cluster, n_init=10)
                labels, indices = self.get_labels_and_indices(data, kmeans, size_sample, num_bootstrap_samples, random_state)    
                
                if self.stability_args.adjusted_rand_score: worker1 = Process(target=self.adjusted_rand_score, args=(labels, indices, cluster, stab_methods))
                if self.stability_args.adjusted_mutual_info_score: worker2 = Process(target=self.adjusted_mutual_info_score, args=(labels, indices, cluster, stab_methods))
                if self.stability_args.bagclust: worker3 = Process(target=self.bagclust, args=(rData, num_bootstrap_samples, cluster, stab_methods))
                if self.stability_args.han: worker4 = Process(target=self.han, args=(rData, num_bootstrap_samples, cluster, stab_methods))
                if self.stability_args.OTstab: worker5 = Process(target=self.OTstab, args=(rData, num_bootstrap_samples, cluster, stab_methods))

                if self.stability_args.adjusted_rand_score: worker1.start()
                if self.stability_args.adjusted_mutual_info_score: worker2.start()
                if self.stability_args.bagclust: worker3.start()
                if self.stability_args.han: worker4.start()
                if self.stability_args.OTstab:  worker5.start()

                if self.stability_args.adjusted_rand_score: worker1.join()
                if self.stability_args.adjusted_mutual_info_score: worker2.join()
                if self.stability_args.bagclust: worker3.join()
                if self.stability_args.han: worker4.join()
                if self.stability_args.OTstab:  worker5.join()
            
            #To JSON serialize 
            stab_methods = dict(stab_methods)
            for k in stab_methods.keys():
                stab_methods[k] = list(stab_methods[k])
        
            stab_epochs[ep] = stab_methods
            
            wan_ls = []
            for k,v in enumerate(clusters):
                if self.stability_args.adjusted_rand_score: wan_ls.append(ast.literal_eval(f"{{'adjusted_rand_score: k({v})': {stab_methods['adjusted_rand_score'][k]}}}"))
                if self.stability_args.adjusted_mutual_info_score: wan_ls.append(ast.literal_eval(f"{{'adjusted_mutual_info_score: k({v})': {stab_methods['adjusted_mutual_info_score'][k]}}}"))
                if self.stability_args.bagclust: wan_ls.append(ast.literal_eval(f"{{'bagclust: k({v})': {stab_methods['bagclust'][k]}}}"))
                if self.stability_args.han: wan_ls.append(ast.literal_eval(f"{{'han: k({v})': {stab_methods['han'][k]}}}"))
                if self.stability_args.OTstab: wan_ls.append(ast.literal_eval(f"{{'OTclust: k({v})': {stab_methods['OTclust'][k]}}}"))
            
            if self.stability_args.report_to: wandb.log(dict(ChainMap(*wan_ls)))
            
            stab_methods = manager.dict()
            stab_methods["adjusted_rand_score"] = manager.list()
            stab_methods["adjusted_mutual_info_score"] = manager.list()
            stab_methods["bagclust"] = manager.list()
            stab_methods["han"] = manager.list()
            stab_methods["OTclust"] = manager.list()
            
        self.save(stab_epochs, arguments, path_and_file_name_to_save)
        if self.stability_args.report_to: wandb.finish()

    def get_labels_and_indices(self, data, clrt_algorithm, size_sample, num_bootstrap_samples, random_state):
        labels = []
        indices = []
        for _ in range(num_bootstrap_samples):
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
    def adjusted_rand_score(self, labels, indices, cluster, stab_methods):
        logger.info("Computing [adjusted_rand_score]")
        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                in_both = np.intersect1d(i, j)
                scores.append(adjusted_rand_score(l[in_both], k[in_both])) 
        stab_methods['adjusted_rand_score'].append(np.mean(scores))

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    def adjusted_mutual_info_score(self, labels, indices, cluster, stab_methods):
        logger.info("Computing [adjusted_mutual_info_score]")
        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                in_both = np.intersect1d(i, j)
                scores.append(adjusted_mutual_info_score(l[in_both], k[in_both]))
        stab_methods['adjusted_mutual_info_score'].append(np.mean(scores))

    #"A prediction-based resampling method for estimating the number of clusters in a dataset."
    #"Bagging to improve the accuracy of a clustering procedure."
    #Explicação: Stability estimation for unsupervised clustering: A review
    def bagclust(self, rData, num_bootstrap_samples, n_cluster, stab_methods):
        logger.info("Computing [bagclust]")
        rDataStab = clue.cl_bag(x = rData, B = num_bootstrap_samples, k = n_cluster)
        stab_methods['bagclust'].append(rDataStab.rx2['.Data'].max(axis = 1).mean())

    #Bootstrapping estimates of stability for clusters,observations and model selection
    #Para entender vá para a página 4 Seção 2 Fig. 1.
    def han(self, rData, num_bootstrap_samples, n_cluster, stab_methods):
        logger.info("Computing [han]")
        try:
            hanStab = bootcluster.stability(x = rData, k = n_cluster, B = num_bootstrap_samples)
            stab_overall = float(0) if math.isnan(float(hanStab.rx2['overall'])) else float(hanStab.rx2['overall'])
            stab_methods['han'].append(stab_overall)
        except RRuntimeError:
            stab_methods['han'].append(float(0))

    #CPS Analysis for cluster validation
    #Install from github https://github.com/cran/OTclust
    #Melhor explicação: Denoising Methods for Inferring Microbiome Community Content and Abundance
    def OTstab(self, rData, num_bootstrap_samples, n_cluster,stab_methods): 
        logger.info("Computing [OTstab]")
        otclust = OTclust.clustCPS(rData, k=n_cluster, l= False, pre=False, noi="after",
                 nPCA = 2, nEXP = num_bootstrap_samples)
        stab_methods['OTclust'].append(float(otclust.rx2['tight_all']))

    def save(self, stab_epochs, arguments, path_and_file_name_to_save):
        data = json.dumps([stab_epochs, arguments], indent = 4)
        i = 1
        path_and_file_name_to_save = path_and_file_name_to_save.replace(".json", "")
        while os.path.exists(f"{path_and_file_name_to_save}-{i}.json"):
            i += 1
        file = open(f"{path_and_file_name_to_save}-{i}.json","w")
        file.write(data)
        file.close()