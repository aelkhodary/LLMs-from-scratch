Introduction
Attention mechanisms are a key component of Large Language Models (LLMs), enabling the model to incorporate context and relationships between different parts of the input sequence.

Self-Attention Mechanism 1

The self-attention mechanism is a crucial component of LLMs, allowing the model to incorporate context and relationships between different parts of the input sequence. It computes the attention weights by relating different positions within a single input sequence, assessing and learning the relationships and dependencies between various parts of the input itself. The self-attention mechanism can be implemented with trainable weights, enabling the model to learn and produce "good" context vectors.

Causal Attention 1

Causal attention is a specialized form of self-attention that restricts the model to only consider previous and current inputs in a sequence when processing any given token. This is achieved by masking out the attention weights above the diagonal, and normalizing the non-masked attention weights such that the attention weights sum to 1 in each row.

Multi-Head Attention 12

Multi-head attention is an extension of single-head attention, where multiple instances of the self-attention mechanism are stacked on top of each other, allowing the model to capture various aspects of the input data in parallel. This is achieved by creating multiple instances of the self-attention mechanism, each with its own weights, and then combining their outputs.

Calculating Attention 23

The attention mechanism is calculated by computing the relevance of each input token to the current token being processed, and then combining the information from the various positions into a single output vector. This is achieved by computing the dot product between the current input element and another element in the input sequence, and then normalizing the weights using the softmax function.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Compute attention weights
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        attention_weights = torch.matmul(query, key.T) / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Compute context vector
        context = torch.matmul(attention_weights, value)
        return context


        1- Build a Large Language Model (From Scratch)
              By Sebastian Raschka
        2- Hands-On Large Language Models
              By Jay Alammar

        3- Machine Learning with PyTorch and Scikit-Learn
           By Sebastian Raschka      