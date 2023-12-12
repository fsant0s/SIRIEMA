from dataclasses import dataclass, field
from config.definitions import ROOT_DIR
import os

from typing import List

@dataclass
class ComputerStabilityArguments:
    """
    Arguments pertaining to calculate stability.
    """
    
    project_name: str = field(default=None,
                           metadata={'help': 'Project name'})
    
    data_path: str = field(default=None,
                           metadata={'help': 'the path to the csv files containing the dataset'})

    output_dir: str = field(default=None,
                           metadata={'help': 'the path to save the results'})
    
    output_stab_name: str = field(default=None,
                           metadata={'help': 'the file to save the results'})
    
    clusters: List[int] = field(default_factory=lambda: [2],
                           metadata={'help': 'List of k(s) to stability evaluate.'})

    num_train_epochs: int = field(default=3,
                           metadata={'help': 'total number of iterations of all the data in one cycle for compute the stability'})

    num_bootstrap_samples: int = field(default=50,
                           metadata={'help': 'number of bootstrap re-samplings'})
    
    num_random_samples: int = field(default=0,
                           metadata={'help': 'number of samples. To use the whole dataset set it to zero'})
    
    repeat_experimet: int = field(default=1,
                           metadata={'help': 'Number of experiments to be done'})
    
    output_log_name: str = field(default="logs.json",
                           metadata={'help': 'output log name'})
    
    mode: str = field(default="CPU",
                           metadata={'help': 'clusters to compute stability'})

    adjusted_rand_score: bool = field(default=True,
                           metadata={'help': 'if true, compute adjusted_rand_score stability. Otherwise, do not.'})

    adjusted_mutual_info_score: bool = field(default=True,
                           metadata={'help': 'if true, compute adjusted_mutual_info_score stability. Otherwise, do not.'})

    bagclust: bool = field(default=True,
                           metadata={'help': 'if true, compute bagclust stability. Otherwise, do not.'})

    han: bool = field(default=True,
                           metadata={'help': 'if true, compute han stability. Otherwise, do not.'})

    OTstab: bool = field(default=True,
                           metadata={'help': 'if true, compute otclust stability. Otherwise, do not.'})
    
    RNDN: int = field(default=1,
                           metadata={'help': 'Random seed used to initialize the pseudo-random number generator or an instantized BitGenerator'})

    report_to: bool = field(default=False,
                        metadata={'help': 'report to wandb or not'})
    