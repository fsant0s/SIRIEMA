import sys
sys.path.append('/customer-segmentation-analysis/') # to run on recod clusters

import os
import pandas as pd
import time
import logging

from config.definitions import ROOT_DIR
sys.path.append(ROOT_DIR)

from src.algorithms.stability.stability_algorithms_cpu import StabilityCPU
#from src.algorithms.stability.stability_algorithms_gpu import StabilityGPU

from utils.util import create_dir_if_not_exists, get_args_info_as_str

from config.definitions import ROOT_DIR

from utils.compute_stability_exp_args import ComputerStabilityArguments

import traceback
from transformers import HfArgumentParser

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
    ComputerStabilityArguments
    )
    try:
        stability_args = parser.parse_json_file(
            json_file= ROOT_DIR + "/datasets/processed/" + sys.argv[1]
        )[0]
    except Exception as e:
        logging.error(traceback.format_exc())
    
    # Setup logging
    create_dir_if_not_exists(ROOT_DIR + stability_args.output_dir)
   
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(
        filename=os.path.join(ROOT_DIR + stability_args.output_dir, stability_args.output_log_name),  encoding='utf-8', mode="w+"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[stream_handler, file_handler],
    )
    
    logger.info(f"======== Stability Args ========\n{get_args_info_as_str(stability_args)}\n")

    st = time.time()
    for i in range(1, stability_args.repeat_experimet + 1):
        data = pd.read_csv(ROOT_DIR + stability_args.data_path, skiprows = 0)
        if stability_args.num_random_samples:
            data = data.sample(n=stability_args.num_random_samples)
        data = data.to_numpy()

        logger.info(f"======== Experiment number: {i} ========")
        logger.info(f"data shape: {data.shape}")
        
        stabO = StabilityCPU(stability_args, ROOT_DIR) #TODO: Pass ROOT_DIR from stability_args
        stabO.run(data)

        logger.info(f"======== Experiment {i} has been done ========\n")
    et = time.time()
    elapsed_time = et - st
    logger.info(f"Execution time: {elapsed_time} seconds")
    
if __name__ == '__main__':
    main()
