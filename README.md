### Communication-Efficient Federated Neural Collaborative Filtering with Multi-Armed Bandits
___
This repository contains the code, dataset, and related guidelines for the practical implementation of our work **Communication-Efficient Federated Neural Collaborative Filtering with Multi-Armed Bandits**. 

We implement the proposed solution with PyTorch. The architecture of the neural collaborative model is made up of two sub-networks: (1) multi-layer perceptron (MLP) and (2) generalized matrix factorization (GMF). Both MLP and GMF networks are used to learn the latent factors in the form of user and item embeddings. We employ a 16-hidden layer for MLP with a ReLU activation function. While restricting the size of the latent factors to 4, we set the architecture of the NCF model to be [ 8, 16, 8 ], where the first and the last layer is the concatenation MLP and GMF individual layers. 
