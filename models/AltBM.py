import torch
import numpy as np
import scipy
from src.utils import sampling_bernoulli, sampling_gaussian
import pdb
class AltBM():#Adversarial Boltzmann Machine
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1


        self.vis_weights = torch.randn(num_visible, num_visible) * 0.1
        self.hid_weights = torch.randn(num_hidden, num_hidden) * 0.1

        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.vis_weights_momentum = torch.zeros(num_visible, num_visible)
        self.hid_weights_momentum = torch.zeros(num_hidden, num_hidden)

        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.vis_weights = self.vis_weights.cuda()
            self.hid_weights = self.hid_weights.cuda()

            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.vis_weights_momentum = self.vis_weights_momentum.cuda()
            self.hid_weights_momentum = self.hid_weights_momentum.cuda()


            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities,size=64):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + torch.matmul(self.hidden,self.hid_weights) + self.hidden_bias #VW+B
        hidden_activations = sampling_bernoulli(hidden_activations) #VW+B
        hidden_probabilities = self._sigmoid(hidden_activations) #sigmoid (VW+B)
        self.hidden = hidden_probabilities
        return hidden_probabilities #output

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) +torch.matmul(self.visible,self.vis_weights) + self.visible_bias #hW_T+C
        visible_activations = sampling_bernoulli(visible_activations)#hW_T+C
        visible_activations = sampling_gaussian(visible_activations)#hW_T+C
        visible_probabilities = self._sigmoid(visible_activations) #Sigmoid (hW_T+c)
        self.visible = visible_probabilities
        return visible_probabilities #output

    def contrastive_divergence(self, input_data):
        # Positive phase
        self.visible = input_data
        batch_size= input_data.size(0)
        self.hidden = torch.randn(batch_size,self.num_hidden)* 0.1
        self.visible = self.visible.cuda()
        self.hidden = self.hidden.cuda()
        positive_hidden_probabilities = self.sample_hidden(input_data)  #up
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float() #Noise sampleing?
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations) #Back_down?
        vis_positive_associations = torch.matmul(input_data.t(), self.visible) #Back_down?
        hid_positive_associations = torch.matmul(positive_hidden_activations.t(), self.hidden) #Back_down?

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)
        vis_negative_associations = torch.matmul(negative_visible_probabilities.t(), self.visible)
        hid_negative_associations = torch.matmul(negative_hidden_probabilities.t(), self.hidden)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)
        self.vis_weights_momentum *= self.momentum_coefficient
        self.vis_weights_momentum += (vis_positive_associations - vis_negative_associations)
        self.hid_weights_momentum *= self.momentum_coefficient
        self.hid_weights_momentum += (hid_positive_associations - hid_negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)



        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.vis_weights += self.vis_weights_momentum * self.learning_rate / batch_size
        self.hid_weights += self.hid_weights_momentum * self.learning_rate / batch_size

        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay
        self.vis_weights -= self.vis_weights * self.weight_decay  # L2 weight decay
        self.hid_weights -= self.hid_weights * self.weight_decay  # L2 weight decay

        #self.weights_visible -= self.weights_visible * self.weight_decay  # L2 weight decay
        #self.weights_hidden -= self.weights_hidden * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)
        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def output(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        #hidden_activations = torch.matmul(hidden_activations, self.hid_weights) + self.hidden_bias  # VW+B
        hidden_activations = sampling_bernoulli(hidden_activations) #VW+B
        hidden_probabilities = self._sigmoid(hidden_activations) #sigmoid (VW+B)
        return hidden_probabilities #output



    def reconstruction(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias #hW_T+C
        visible_probabilities = self._sigmoid(visible_activations)  # Sigmoid (hW_T+c)
        return visible_probabilities #output