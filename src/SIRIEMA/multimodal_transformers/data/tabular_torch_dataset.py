import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class TorchTabularTextDataset(TorchDataset):
    """
    :obj:`TorchDataset` wrapper for text dataset with categorical features
    and numerical features

    Parameters:
        encodings (:class:`transformers.BatchEncoding`):
            The output from encode_plus() and batch_encode() methods (tokens, attention_masks, etc) of
            a transformers.PreTrainedTokenizer
        categorical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, categorical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed categorical features
        numerical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, numerical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed numerical features
        labels (:class: list` or `numpy.ndarray`, `optional`, defaults to :obj:`None`):
            The labels of the training examples
        class_weights (:class:`numpy.ndarray`, of shape (n_classes),  `optional`, defaults to :obj:`None`):
            Class weights used for cross entropy loss for classification
        df (:class:`pandas.DataFrame`, `optional`, defaults to :obj:`None`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            TabularConfig instance specifying the configs for TabularFeatCombiner

    """

    def __init__(
        self,
        encodings,
        categorical_feats,
        numerical_feats,
        df=None,
        class_weights=None,
    ):
        self.df = df
        self.encodings = encodings
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.class_weights = class_weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        item["cat_feats"] = (
            torch.tensor(self.cat_feats[idx]).float()
            if self.cat_feats is not None
            else torch.zeros(0)
        )
        item["numerical_feats"] = (
            torch.tensor(self.numerical_feats[idx]).float()
            if self.numerical_feats is not None
            else torch.zeros(0)
        )
        return item

    def __len__(self):
        return int(self.df.shape[0])


