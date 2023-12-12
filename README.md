
# SIRIEMA: A Framework to Enhance Clustering Stability by Fusing Multimodal Data

- **Problem Addressed**: This work aims to tackle the clustering stability issue. We introduce SIRIEMA, a novel multimodal framework that seamlessly integrates categorical, numerical, and text data to bolster clustering robustness.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1PuMogHiBI5-phceoKn8UZM7CcUlLOFKa"/>
</div>

- **Versatility of the Framework**: Our approach can employ any model that generates embeddings. This encompasses a wide range of transformer-based models.

- **Embedding Integration**: The chosen embedding model serves as the bedrock for processing text features and can be easily integrated into our framework.

# Multimodal Integration Component

- **Unique Multimodal Integration**: Our framework boasts an exclusive integration module. This module blends the output from the embedding model with categorical and numerical features, crafting unique multimodal attributes.

<div align="center"><img src="https://drive.google.com/uc?export=view&id=1vLO-9I1sVKhw9h0683t3hdL59rtcQzx-"/></div>

> **Note**: Uppercase bold letters represent 2D matrices, and lowercase bold letters represent 1D vectors. **b** is a scalar bias, **W** represents a weight matrix.

- **Utilization of Generative Models**: These enriched features are further processed by generative models like VAE (Variational Autoencoder) or GAN (Generative Adversarial Network). This ensures clusters are cohesive, reduces variance, and guarantees more stable outcomes


# How to run:

## Install R and Python:

### R
- R.version: 4.2.2 https://cran.r-project.org/bin/windows/base/old/4.2.2/
- Rtools: https://cran.r-project.org/bin/windows/Rtools/rtools42/rtools.html

install.packages("tsne")
install.packages("devtools")
install.packages("dplyr")
install.packages("clue")
install.packages("devtools")
library(devtools)
install_github("cran/OTclust")

### Python (any version):
- pyenv local 3.10.0
- python -m venv . (inside customer-segmentation-analysis folder)
- /Script/activated
- python -m pip install -r requirements.txt