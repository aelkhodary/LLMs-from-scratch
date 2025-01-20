# Chapter 3: Coding Attention Mechanisms

&nbsp;
## Main Chapter Code

- [01_main-chapter-code](01_main-chapter-code) contains the main chapter code.

&nbsp;
## Bonus Materials

- [02_bonus_efficient-multihead-attention](02_bonus_efficient-multihead-attention) implements and compares different implementation variants of multihead-attention
- [03_understanding-buffers](03_understanding-buffers) explains the idea behind PyTorch buffers, which are used to implement the causal attention mechanism in chapter 3



# Self-Attention Mechanism in LLMs

This README explains how the attention mechanism works in Large Language Models (LLMs) like GPT, with a mathematical example.

### Key Points:
- The attention mechanism calculates weights to determine how much each token contributes to the understanding of another token.
- These weights are used to create context vectors, which are passed to the next layers of the LLM for further processing.
- The result ensures the model focuses on relevant parts of the input for accurate predictions.
- This process forms the backbone of attention mechanisms in modern LLMs like GPT.

---

## Steps in Self-Attention Calculation
Self-Attention Mechanism in LLMs

1. Input Context
   - Each token is represented as a vector.
     - "The": [1, 0, 1]
     - "cat": [0, 1, 1]
     - "sat": [1, 1, 0]

2. Generate Queries (Q), Keys (K), and Values (V)
 
   - Queries (Q) = Input × W_Q
     - W_Q = [[1, 0], [0, 1], [1, 1]]
   - Keys (K) = Input × W_K
     - W_K = [[0, 1], [1, 0], [1, 1]]
   - Values (V) = Input × W_V
     - W_V = [[1, 1], [0, 1], [1, 0]]

3. Compute Attention Scores
   - Dot product of Queries and Keys.
   Score(i, j) = Q_i · K_j
   - For "The" (Query):
     - Score(The, The): 4
     - Score(The, cat): 5
     - Score(The, sat): 1
   - Score(The, The): Q_The · K_The = [2, 1] · [1, 2] = 2 × 1 + 1 × 2 = 4
   - Score(The, cat): Q_The · K_cat = [2, 1] · [2, 1] = 2 × 2 + 1 × 1 = 5
   - Score(The, sat): Q_The · K_sat = [2, 1] · [0, 1] = 2 × 0 + 1 × 1 = 1

   The raw scores for "The" are: [4, 5, 1]


4. Apply Softmax
   - Convert scores to probabilities.
     - Attention Weights: [0.27, 0.72, 0.01]

5. Compute Context Vector
   - Weighted sum of Values.
     - Context Vector for "The": [1.27, 1.01]
---

### **Final Output**:
- **Input**: `[1, 0, 1], [0, 1, 1], [1, 1, 0]`
- **Context Vector for "The"**: `[1.27, 1.01]`

---

## **Key Points**
1. The attention mechanism calculates **weights** to determine how much each token contributes to the understanding of another token.
2. These weights are used to create **context vectors**, which are passed to the next layers of the LLM for further processing.
3. The result ensures the model focuses on **relevant parts** of the input for accurate predictions.

This process forms the backbone of attention mechanisms in modern LLMs like GPT.

### Summary:
- **Input**: Token vectors.
- **Output**: Context vectors (e.g., [1.27, 1.01] for "The").
- The context vector is computed by:
  1. Generating Queries, Keys, and Values.
  2. Calculating attention scores via dot products.
  3. Applying Softmax to compute weights.
  4. Using weights to create a weighted sum of Value vectors.
  
This process allows LLMs to focus on relevant parts of the input for better context understanding!
 

 # 3.4.1 Computing the attention weights step by step