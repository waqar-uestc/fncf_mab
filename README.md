### Communication-Efficient Federated Neural Collaborative Filtering
___

This repository contains the code, dataset, and related guidelines for the practical implementation of our work **Communication-Efficient Federated Neural Collaborative Filtering with Multi-Armed Bandits**. 

## Introduction
Federated learning (FL) has received much attention in privacy-preserving and responsible recommender systems. Recent studies have shown promising results while federating widely-used recommendation methods such as collaborative filtering. A major barrier when bringing FL into production is that the model complexity or the volume of gradients to be transmitted over the communication channel grows linearly as the number of items in a particular system increases. To address this challenge, we propose a communication-efficient neural collaborative filtering method for federated recommender systems. 

We implement the proposed solution with PyTorch. The architecture of the neural collaborative model is made up of two sub-networks: (1) multi-layer perceptron (MLP) and (2) generalized matrix factorization (GMF). Both MLP and GMF networks are used to learn the latent factors in the form of user and item embeddings. We employ a 16-hidden layer for MLP with a ReLU activation function. While restricting the size of the latent factors to 4, we set the architecture of the NCF model to be [ 8, 16, 8 ], where the first and the last layer is the concatenation MLP and GMF individual layers. 
___

## Usage
First, You need to install the required dependencies:  
```
pip install -r requirements.txt 
```


For simple neural collaborative filtering run: 
```bash
python ncf_ml1m.py 
```
To run federated version of neural collaborative filtering run: 
```bash
python fncf_ml1m.py 
```
For bandit based communication-efficient neural collaborative filtering run: 
```bash
python fncf_mab_ml1m.py 
```
## Support
If you have any questions, feel free to contact us for assistance ! 
