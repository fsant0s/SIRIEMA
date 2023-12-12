class TabularConfig:
    r"""Config used for tabular combiner

    Args:
        vae_division (int): how much to decrease each VAE dim for each additional layer.
        combine_feat_method (str): The method to combine categorical and numerical features.
            See :obj:`TabularFeatCombiner` for details on the supported methods.
        vae_dropout (float): dropout ratio used for VAE layers
        numerical_bn (bool): whether to use batchnorm on numerical features
        use_simple_classifier (bool): whether to use single layer or VAE as final classifier
        vae_act (str): the activation function to use for finetuning layers
        gating_beta (float): the beta hyperparameters used for gating tabular data
            see the paper `Integrating Multimodal Information in Large Pretrained Transformers <https://www.aclweb.org/anthology/2020.acl-main.214.pdf>`_ for details
        numerical_feat_dim (int): the number of numerical features
        cat_feat_dim (int): the number of categorical features

    """

    def __init__(
        self,
        vae_out_dim,
        vae_division=2,
        mlp_division=4,
        latent_dim=3,
        bn_enc=False,
        bn_dec=False,
        combine_feat_method="text_only",
        vae_dropout=0.1,
        mlp_dropout=0.1,
        numerical_bn=True,
        use_simple_classifier=True,
        vae_act="relu",
        mlp_act="relu",
        gating_beta=0.2,
        numerical_feat_dim=0,
        cat_feat_dim=0,
        **kwargs
    ):
        self.latent_dim = latent_dim,
        self.bn_enc = bn_enc,
        self.bn_dec = bn_dec,
        self.vae_division = vae_division
        self.mlp_division = mlp_division
        self.combine_feat_method = combine_feat_method
        self.vae_dropout = vae_dropout
        self.mlp_dropout = mlp_dropout
        self.numerical_bn = numerical_bn
        self.use_simple_classifier = use_simple_classifier
        self.vae_act = vae_act
        self.mlp_act = mlp_act
        self.gating_beta = gating_beta
        self.numerical_feat_dim = numerical_feat_dim
        self.cat_feat_dim = cat_feat_dim
        self.vae_out_dim = vae_out_dim
