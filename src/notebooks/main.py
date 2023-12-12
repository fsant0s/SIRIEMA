import logging
import os
from statistics import mean, stdev
import sys

import pandas as pd
import wandb
import torch
from pprint import pformat

from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback
)

from multimodal_exp_args import (
    MultimodalDataTrainingArguments,
    ModelArguments,
    OurTrainingArguments,
    ComputerStabilityArguments,
)
from evaluation import calc_stability_metrics
from multimodal_transformers.data import load_data_from_folder, load_data_into_folds
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import CustomTrainer
from util import create_dir_if_not_exists, get_args_info_as_str

os.environ["COMET_MODE"] = "DISABLED"
logger = logging.getLogger(__name__)

class CustomCallback(TrainerCallback):
    def __init__(self, trainer, stability_args, wandb):
        super().__init__()
        self.trainer = trainer
        self.stability_args = stability_args
        self.wandb = wandb

    def calc_stability(self, coming_from_train_end = False):
        if (self.stability_args.compute_stability_steps > 0 
            and self.state.global_step % self.stability_args.compute_stability_steps == 0) or coming_from_train_end:

            val_outputs = self.trainer.predict(self.trainer.eval_dataset).predictions
            stability_out = calc_stability_metrics(
                val_outputs, 
                self.stability_args.clusters, 
                self.stability_args.k_means_random_state,
                self.stability_args.n_samples, 
                self.stability_args.randomStateSeed
            )
            logger.info(f"*** Stability results *** : \n{stability_out}\n")
            # Log to wandb
            if self.wandb:
                self.wandb.log(stability_out)

    def on_step_end(self, args, state, control, **kwargs):
        self.calc_stability()

    def on_train_end(self, args, state, control, **kwargs):
        self.calc_stability(coming_from_train_end = True)
               
def main():
    parser = HfArgumentParser(
        (ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments, ComputerStabilityArguments)
    )
    model_args, data_args, training_args, stability_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    create_dir_if_not_exists(training_args.output_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(
        filename=os.path.join(training_args.output_dir, "train_log.txt"),  encoding='utf-8', mode="w+"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[stream_handler, file_handler],
    )

    logger.info(f"======== Model Args ========\n{get_args_info_as_str(model_args)}\n")
    logger.info(f"======== Data Args ========\n{get_args_info_as_str(data_args)}\n")
    logger.info(f"======== Training Args ========\n{get_args_info_as_str(training_args)}\n")
    logger.info(f"======== Stability Args ========\n{get_args_info_as_str(stability_args)}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if not data_args.create_folds:
        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            data_args.train_csv_name, 
            data_args.val_csv_name, 
            data_args.test_csv_name,
            data_args.num_random_samples,
            data_args.data_path,
            data_args.column_info["text_cols"],
            tokenizer,
            categorical_cols=data_args.column_info["cat_cols"],
            numerical_cols=data_args.column_info["num_cols"],
            categorical_encode_type=data_args.categorical_encode_type,
            numerical_transformer_method=data_args.numerical_transformer_method,
            sep_text_token_str=tokenizer.sep_token
            if not data_args.column_info["text_col_sep_token"]
            else data_args.column_info["text_col_sep_token"],
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
            debug_dataset_size=training_args.debug_dataset_size,
        )
        train_datasets = [train_dataset]
        val_datasets = [val_dataset]
        test_datasets = [test_dataset]
    else:
        train_datasets, val_datasets, test_datasets = load_data_into_folds(
            data_args.data_path,
            data_args.num_folds,
            data_args.validation_ratio,
            data_args.column_info["text_cols"],
            tokenizer,
            categorical_cols=data_args.column_info["cat_cols"],
            numerical_cols=data_args.column_info["num_cols"],
            categorical_encode_type=data_args.categorical_encode_type,
            numerical_transformer_method=data_args.numerical_transformer_method,
            sep_text_token_str=tokenizer.sep_token
            if not data_args.column_info["text_col_sep_token"]
            else data_args.column_info["text_col_sep_token"],
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
            debug_dataset_size=training_args.debug_dataset_size,
        )
    train_dataset = train_datasets[0]
    set_seed(training_args.seed)
      
    # set the wandb project where this run will be logged
    wandb.init(project = training_args.experiment_name)

    total_results = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(
        zip(train_datasets, val_datasets, test_datasets)
    ):
        logger.info(f"======== Fold {i+1} ========")
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        tabular_config = TabularConfig(
            cat_feat_dim=train_dataset.cat_feats.shape[1]
            if train_dataset.cat_feats is not None
            else 0,
            numerical_feat_dim=train_dataset.numerical_feats.shape[1]
            if train_dataset.numerical_feats is not None
            else 0,
            **vars(data_args),
        )
        config.tabular_config = tabular_config
        model = AutoModelWithTabular.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if i == 0:
            logger.info(tabular_config)
            logger.info(model)

        trainer = CustomTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            compute_metrics = None, #evaluation strategy to adopt during training calling by evaluation_strategy. See bertvaewithtabular/multimodal_transformers/model/custom_trainer.py#L261
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.add_callback(CustomCallback(trainer, stability_args, wandb))

       # To freeze all transformer weights
        if data_args.freeze_transformer_weights:        
            for param in model.embeddings.parameters():
                param.requires_grad = False
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.pooler.parameters():
                param.requires_grad = False

        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.save_model()
            
        logger.info("========  Training has finished ========\n")
        #train_results = trainer.evaluate(eval_dataset=train_dataset,  metric_key_prefix = "train")

        # Evaluation
        eval_results = {}
        if training_args.do_eval:
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_metric_results_fold_{i+1}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

        # Testing
        if training_args.do_predict:
            logger.info("*** Representation predictions for test dataset ***")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            csv_name = training_args.experiment_name.replace(" ", "")
            output_test_file = os.path.join(
                training_args.output_dir, f"{csv_name}_fold_{i+1}.csv"
            )
            if trainer.is_world_process_zero():
                pd.DataFrame(predictions).to_csv(output_test_file, index = False)

        del model
        del config
        del tabular_config
        del trainer
        torch.cuda.empty_cache()
        total_results.append(eval_results)
        #total_results.append(train_results)

    aggr_res = aggregate_results(total_results)
    output_aggre_test_file = os.path.join(
        training_args.output_dir, f"all_test_metric_results.txt"
    )
    with open(output_aggre_test_file, "w") as writer:
        logger.info("***** Aggr results *****")
        for key, value in aggr_res.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

    wandb.finish()

def aggregate_results(total_test_results):
    metric_keys = list(total_test_results[0].keys())
    aggr_results = dict()

    for metric_name in metric_keys:
        if type(total_test_results[0][metric_name]) is str:
            continue
        res_list = []
        for results in total_test_results:
            res_list.append(results[metric_name])
        if len(res_list) == 1:
            metric_avg = res_list[0]
            metric_stdev = 0
        else:
            metric_avg = mean(res_list)
            metric_stdev = stdev(res_list)

        aggr_results[metric_name + "_mean"] = metric_avg
        aggr_results[metric_name + "_stdev"] = metric_stdev
    return aggr_results

if __name__ == "__main__":
    main()
