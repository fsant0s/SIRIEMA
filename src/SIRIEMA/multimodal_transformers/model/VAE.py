import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import MSELoss

import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from torch.distributions import Normal, Categorical

'''
This is a variational autoencoder (VAE) with {dims} hidden layers

Training
- Start by iterating over the dataset
- During each iteration, pass the data to the encoder to obtain a set of mean and log-variance parameters of the approximate posterior q(z|x)
- then apply the reparameterization trick to sample from q(z|x)
Finally, pass the reparameterized samples to the decoder to obtain the logits of the generative distribution p(x|z)

Read more: 
- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
- https://www.tensorflow.org/tutorials/generative/cvae#network_architecture
- https://github.com/nhsx/SynthVAE/blob/main/VAE.py
'''
'''
#TODO Add clustering loss
# https://naserian-elahe.medium.com/deep-embedding-and-clustering-an-step-by-step-python-implementation-bd2c9d51c80f
Improved Deep Embedded Clustering with Local Structure Preservation: https://www.ijcai.org/proceedings/2017/0243.pdf
- The biggest contribution of DEC is the clustering loss (or target distribution P, to be specific)
- https://github.com/piiswrong/dec
'''
def vae_hf_loss_func(inputs, classifier):
    """
    Computes VAE loss.
    :param inputs: original input data
    :param classifier: classifier. Usually is VAE
    :return total_loss: recon_loss + KL divergence
    :return mu: mean from Encoder
    :return [inputs, x_recon]: >> made it clenear later <<
    """
    x_recon, mu, logvar, z, gmm_loss = classifier(inputs)
    
    # Reconstruction loss (using Mean Squared Error for non-binary data)
    recon_loss  = MSELoss(reduction="sum")(x_recon, inputs) 

    # KL divergence
    loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # K-means clustering loss
    kmeans = KMeans(n_clusters=2, n_init=10)  # Define the number of clusters
    kmeans.fit(z.detach().cpu().numpy())
    cluster_loss = torch.tensor(cdist(z.detach().cpu().numpy(), kmeans.cluster_centers_).min(axis=1)[0]).mean()
    
    # Total VAE loss
    total_loss = recon_loss + loss_KLD + gmm_loss

    return total_loss, mu, [inputs, x_recon]

def calc_VAE_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    dims.append(int(dim))
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    dims = dims[:-1]
    return dims[:int(len(dims)/2)]

def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    if act == "lrelu":
        return nn.LeakyReLU(0.1)
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError("Unknown activation function {}".format(act))

'''
Function to get sample after trainning.
TODO: recode to torch.
import tensorflow as tf
@tf.function
def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)
'''

def sample_z(args):
    mu, log_sigma = args
    eps = Variable(torch.randn(mu.size())).to(mu.device)
    return mu + torch.exp(log_sigma * .5) * eps

class Encoder(nn.Module):
    """
        Encoder
        :param dims: Enconder dimension
        :param dropout_prob: Dropout probability
        :param act: Activation function
        :param return_layer_outs: Return layer outputs?
        :param latent_dim: latent dimension size
        :param bn_enc: Bacth normalization?
    """
    def __init__(
        self,
        dims,
        dropout_prob,
        act,
        return_layer_outs,
        latent_dim,
        bn_enc = False
    ):
        super(Encoder, self).__init__()
        self.bn_enc = bn_enc
        self.bn = None
        self.dropout = nn.Dropout(dropout_prob)
        self.act_name = act
        self.encoder_dimensions = dims #+ [dims[-1]] + [dims[-1]]
        self.activation = create_act(act)
        self.return_layer_outs = return_layer_outs

        self.layers = nn.ModuleList(
            list(
                map(
                    self.weights_init_uniform_rule,
                    [
                        nn.Linear(self.encoder_dimensions[i], self.encoder_dimensions[i + 1])
                        for i in range(len(self.encoder_dimensions) - 1)
                    ],
                )
            )
        )

        # distribution parameters
        self.layers.extend(
            nn.ModuleList(
                list(
                    map(
                        self.weights_init_uniform_rule,
                        [
                            nn.Linear(self.encoder_dimensions[-1], latent_dim),
                            nn.Linear(self.encoder_dimensions[-1], latent_dim)
                        ],
                    )
                )
            )
        )
        '''
        Note, it's common practice to avoid using batch normalization when training VAEs, 
        since the additional stochasticity due to using mini-batches may aggravate instability 
        on top of the stochasticity from sampling.
        https://www.tensorflow.org/tutorials/generative/cvae#network_architecture
        '''
        if self.bn_enc:
            self.bn = nn.ModuleList(
                [torch.nn.BatchNorm1d(dim) for dim in self.encoder_dimensions[1:]]
            )
        
    #It doenst work! Weights is NaN
    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m
    
    def weights_init_uniform_rule(self, m, activation=None):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-2]: #Is it the penultimate layer? No activation
                mu = layer(input) # penultimate layer: compute mean
                log_sigma = self.layers[-1](input) #last layer: compute variance
                if self.return_layer_outs:
                    return mu, log_sigma, layer_inputs
                return mu, log_sigma
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

class Decoder(nn.Module):
    """
        Decoder
        :param dims: Enconder dimension
        :param dropout_prob: Dropout probability
        :param act: Activation function
        :param bn: Bacth normalization?
        :param return_layer_outs: Return layer outputs?
    """
    def __init__(
        self,
        dims,
        dropout_prob,
        act,
        return_layer_outs,
        bn_dec = False,
    ):
        super(Decoder, self).__init__()
        self.bn_dec = bn_dec
        self.bn = None
        self.dropout = nn.Dropout(dropout_prob)
        self.act_name = act
        self.decoder_dimensions = dims[::-1] #+ [dims[::-1][-1]] 
        self.activation = create_act(act)
        self.return_layer_outs = return_layer_outs
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weights_init_uniform_rule,
                    [
                        nn.Linear(self.decoder_dimensions[i], self.decoder_dimensions[i + 1])
                        for i in range(len(self.decoder_dimensions) - 1)
                    ],
                )
            )
        )
        '''
        Note, it's common practice to avoid using batch normalization when training VAEs, 
        since the additional stochasticity due to using mini-batches may aggravate instability 
        on top of the stochasticity from sampling.
        https://www.tensorflow.org/tutorials/generative/cvae#network_architecture
        '''
        if self.bn_dec:
            self.bn = nn.ModuleList(
                [torch.nn.BatchNorm1d(dim) for dim in self.decoder_dimensions[1:]]
            )
    
    "It doenst work!"
    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m
    
    def weights_init_uniform_rule(self, m, activation=None):
         n = m.in_features
         y = 1.0/np.sqrt(n)
         m.weight.data.uniform_(-y, y)
         m.bias.data.fill_(0)
         return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]: # is it the last layer? No activation
                output = layer(input)
                layer_inputs.append(output)
                if self.return_layer_outs:
                    return output, layer_inputs
                return output
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

#TODO: A Gaussian Mixture Variational Autoencoder 
class VAE(nn.Module):
    def __init__(
        self,
        dims,
        dropout_prob,
        act,
        return_layer_outs,
        latent_dim,
        bn_enc,
        bn_dec
        ):
        super(VAE, self).__init__()

        self.n_clusters = 2
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            dims = dims, 
            dropout_prob = dropout_prob, 
            act = act,
            return_layer_outs = return_layer_outs, 
            latent_dim = latent_dim,
            bn_enc = bn_enc
        )
        self.decoder = Decoder(
            dims = dims, 
            dropout_prob = dropout_prob,
            act = act, 
            return_layer_outs = return_layer_outs, 
            bn_dec = bn_dec
        )
        
        # Initialize the GMM parameters
        self.mu = nn.Parameter(torch.randn(self.n_clusters, latent_dim))
        self.logvar = nn.Parameter(torch.randn(self.n_clusters, latent_dim))
        self.pi = nn.Parameter(torch.randn(self.n_clusters))
    
    def gmm_loss(self, z):
        q = Categorical(torch.softmax(self.pi, dim=0))
        components = Normal(self.mu, torch.exp(0.5*self.logvar))

        log_pi = torch.log_softmax(self.pi, dim=0)
        log_q = components.log_prob(z.unsqueeze(1)).sum(dim=-1) + log_pi

        return -torch.logsumexp(log_q, dim=1).mean()

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = sample_z([mu, log_sigma])
        gmm_loss = self.gmm_loss(z)        
        return self.decoder(z), mu, log_sigma, z, gmm_loss
