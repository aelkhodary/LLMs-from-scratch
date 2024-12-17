# Chapter 2: Working with Text Data

&nbsp;
## Main Chapter Code

- [01_main-chapter-code](01_main-chapter-code) contains the main chapter code and exercise solutions

&nbsp;
## Bonus Materials

- [02_bonus_bytepair-encoder](02_bonus_bytepair-encoder) contains optional code to benchmark different byte pair encoder implementations

- [03_bonus_embedding-vs-matmul](03_bonus_embedding-vs-matmul) contains optional (bonus) code to explain that embedding layers and fully connected layers applied to one-hot encoded vectors are equivalent.

- [04_bonus_dataloader-intuition](04_bonus_dataloader-intuition) contains optional (bonus) code to explain the data loader more intuitively with simple numbers rather than text.


# 2 Working with text data
## Summary: Preparing Data for LLM Training

### Chapter Focus:
This chapter explains how to prepare text data for training large language models (LLMs), specifically focusing on decoder-only transformer-based models like GPT.

### Key Steps Covered:
- **Tokenization:** Splitting text into words and subword tokens.
- **Advanced Tokenization:** Using methods like byte pair encoding (BPE), commonly used in GPT-like models.
- **Sliding Window Sampling:** A technique to create training examples by moving a fixed-size window over the text.
- **Vector Conversion:** Converting tokens into numerical representations (vectors) that can be fed into an LLM.

### Purpose:
Pretraining LLMs involves a next-word prediction task using massive datasets, requiring efficient data preprocessing to ensure high-quality training. Well-prepared data helps models achieve strong capabilities, enabling further fine-tuning for specific tasks.

### Chapter Objective:
Teach how to implement the data pipeline for stage 1 of building an LLM, including tokenization, encoding, and input-output pair sampling. By the end, you'll know how to prepare and process text data for training your own LLM.

# Start Generation Here
![LLM Image](/llm02.png)

# 2.1 Understanding word embeddings

### Key Points: Deep Neural Networks and Embeddings

#### Why Raw Text Can't Be Processed Directly:
- Neural networks, including LLMs, cannot directly process raw text because it is categorical and incompatible with the mathematical operations used in these models.
- Text needs to be converted into continuous-valued vector representations to be usable.

#### What Are Embeddings?:
- Embeddings map discrete objects (e.g., words, sentences, or documents) to points in a continuous vector space.
- This transformation allows neural networks to process nonnumeric data like text, video, and audio.

#### Embedding Models are Data-Specific:
- Different types of data (text, audio, video) require distinct embedding models. For example:
  - A text embedding model cannot handle video data.
- Embeddings are often part of the input layer of neural networks, optimized during training.

![Embedding Model](/llm03.png)

#### Common Text Embedding Techniques:
- Early techniques like Word2Vec generate word embeddings by predicting a word’s context or vice versa.
- Words appearing in similar contexts (e.g., types of birds) are mapped close to each other in the vector space, as shown in figure 2.3.

#### Dimensionality of Embeddings:
- Word embeddings can range from 1 to thousands of dimensions.
- Higher dimensions capture more nuances but are computationally expensive.
  - Example: Small GPT-2 models use 768 dimensions.
  - Large GPT-3 models use 12,288 dimensions.

#### Pretrained vs. Optimized Embeddings:
- Pretrained models like Word2Vec provide generic embeddings.
- LLMs generate and optimize their own embeddings during training, tailored to specific tasks and datasets.

#### Challenges in Visualization:
- High-dimensional embeddings are difficult to visualize (limited to 2D or 3D plots).
- Simplified scatterplots (e.g., figure 2.3) show clusters of similar concepts in a 2D space.

#### Embedding Steps for LLMs:
- Preparing embeddings involves:
  - Splitting text into words.
  - Converting words into tokens.
  - Turning tokens into vector representations (embedding vectors).
- By optimizing embeddings during training, LLMs achieve better performance and task-specific adaptability. Further steps for implementation will follow in the chapter.

-  Sentence or paragraph embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented generation combines generation (like producing text) with retrieval (like searching an external knowledge base) to pull relevant information when generating text, which is a technique that is beyond the scope of this book.

# 2.2 Tokenizing text
### Key Points: Tokenizing Text for LLM Training

1. **Why Tokenization is Needed:**
   - LLMs cannot process raw text directly; input text must be split into smaller units called tokens (words or special characters).
   - Tokenization is a critical preprocessing step for creating embeddings.
![Text Embedding Techniques](/llm04.png)
2. **Example Dataset:**
   - The text used is Edith Wharton’s "The Verdict", a public domain short story.
   - The text file can be loaded using Python utilities or downloaded from the book’s GitHub repository.

3. **Basic Tokenization Steps:**
   - Use Python’s regular expressions (`re`) to split text into tokens:
     - Initial tokenization splits words and spaces.
     - Punctuation is further separated as individual tokens.
   - **Example output:**
     - Input: "Hello, world. This is a test."
     - Tokens: `['Hello', ',', 'world', '.', 'This', 'is', 'a', 'test', '.']`

4. **Handling Special Characters:**
   - Additional adjustments allow handling special punctuation (e.g., question marks, quotation marks, double dashes).
   - **Modified regex example:**
     - Input: "Hello, world. Is this-- a test?"
     - Tokens: `['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']`

5. **Whitespace Handling:**
   - Whitespace characters are typically removed for simplicity but can be retained if the application (e.g., Python code) requires strict spacing.

6. **Application to Full Text:**
   - Applying the tokenizer to "The Verdict" results in 4,690 tokens without whitespaces.
   - Sample tokens include words and punctuation, neatly separated for further processing.

7. **Output Examples:**
   - First 30 tokens of the story:
     - `['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']`

8. **Advanced Tokenization:**
   - This manual tokenization approach works for small datasets but will later transition to prebuilt tokenizers for handling larger and more complex datasets.
   - By the end of this process, the text is fully tokenized and ready for further steps like embedding generation and training.

   ![Text Embedding Techniques](/llm05.png)


# 2.3 Converting tokens into token IDs.

![Text Embedding Techniques](/llm06.png)


# 2.4 Adding special context tokens
### Key Points: Tokenizing Text and Mapping Tokens to IDs

# Start Generation Here
![Text Embedding Techniques](/llm07.png)
![Text Embedding Techniques](/llm08.png)

1. **Purpose of Token IDs:**
   - Tokens (words or special characters) are mapped to unique integers (token IDs) to prepare them for embedding vectors used in LLM training.
   - This involves building a vocabulary that assigns a unique integer to each token.

2. **Creating a Vocabulary:**
   - Tokenize the entire training text into individual tokens.
   - Remove duplicates and sort tokens alphabetically to build a vocabulary.
   - Example: Vocabulary size for Edith Wharton’s "The Verdict" is 1,130 unique tokens.

3. **Tokenization Process:**
   - Use a Python dictionary to map tokens to IDs.
   - Example: `{ "!", 0, '"', 1, "Hermia", 50 }`
   - Token IDs can then be used to encode text for LLM training.

4. **Reverse Mapping:**
   - To convert token IDs back into text, create an inverse vocabulary mapping IDs to tokens.
   - This enables decoding token IDs into their original text form.

5. **Implementing a Tokenizer in Python:**
   - A Python class is built with two main methods:
     - **Encode:** Converts text into token IDs using the vocabulary.
     - **Decode:** Converts token IDs back into text using the inverse vocabulary.
   - Additional preprocessing ensures proper handling of punctuation and spaces.

6. **Example Workflow:**
   - **Encoding:**
     - Input: "It's the last he painted, you know."
     - Output (Token IDs): `[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596]`
   - **Decoding:**
     - Input: Token IDs `[1, 56, 2, 850, 988]`
     - Output: "It's the last he painted"

7. **Challenges with New Words:**
   - If a word is not in the vocabulary (e.g., "Hello"), the tokenizer raises an error.
   - This highlights the need for large and diverse training datasets to ensure comprehensive vocabularies.

8. **Future Enhancements:**
   - Handle unknown tokens gracefully using special tokens (e.g., `<UNK>`).
   - Extend vocabularies with larger, more diverse datasets for broader applicability in LLMs.

9. **Next Steps:**
   - Test the tokenizer further with text containing unknown words.
   - Explore the use of special tokens to provide additional context during LLM training.

# 2.4 Adding special context tokens

### Key Points: Handling Unknown Words and Special Tokens

1. **Why Modify the Tokenizer?**
   - To handle unknown words (not in the vocabulary).
   - To add special tokens that improve context understanding, like markers for boundaries or padding.

2. **Adding Special Tokens:**
   - `<|unk|>`: Represents unknown words not in the vocabulary.
   - `<|endoftext|>`: Separates unrelated text segments (e.g., between books or documents).
   - These tokens are added to the vocabulary and assigned unique IDs.

3. **Implementing the Updated Tokenizer (SimpleTokenizerV2):**
   - Unknown words are replaced with the `<|unk|>` token.
   - Special tokens like `<|endoftext|>` are handled seamlessly during encoding and decoding.
   - **Example:** Input text with boundaries:
     - "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
     - The tokenizer converts it into token IDs where unknown words and boundaries are represented appropriately.

4. **Testing the Tokenizer:**
   - Text containing unknown words ("Hello", "palace") uses `<|unk|>`.
   - The `<|endoftext|>` token separates the two unrelated text samples.
   - Detokenizing successfully reproduces the structure with unknown word placeholders.

5. **Additional Special Tokens:**
   - `[BOS]`: Marks the beginning of a sequence.
   - `[EOS]`: Marks the end of a sequence.
   - `[PAD]`: Adds padding to ensure uniform text length for batch processing.

6. **GPT Tokenizer Simplifications:**
   - GPT models:
     - Use `<|endoftext|>` for both boundary marking and padding.
     - Do not use `<|unk|>`; instead, they use byte pair encoding (BPE) to break unknown words into subword units.
     - This makes GPT’s tokenizer efficient and avoids the need for an explicit unknown token.

7. **Summary:**
   - The updated tokenizer handles unknown words (`<|unk|>`) and boundaries (`<|endoftext|>`), improving its robustness for real-world text.
   - GPT models simplify this further by relying on byte pair encoding tokenizer BPE instead of `<|unk|>`, which ensures all words are tokenized into known subword units.


# 2.5 Byte pair encoding

### Key Points: Byte Pair Encoding (BPE) Tokenization

1. **What is BPE?**
   - BPE is an advanced tokenization method used in models like GPT-2, GPT-3, and the original ChatGPT.
   - It efficiently handles unknown words by breaking them into subwords or individual characters.

2. **Why Use BPE?**
   - It eliminates the need for an `<|unk|>` token because unknown words are split into smaller, known parts.
   - BPE ensures that any word, even unfamiliar ones, can be tokenized and reconstructed.

3. **Using the tiktoken Library:**
   - Install the BPE tokenizer library:
     ```bash
     pip install tiktoken
     ```
   - Instantiate the tokenizer for GPT-2 vocabulary:
     ```python
     tokenizer = tiktoken.get_encoding("gpt2")
     ```

4. **Tokenizing and Decoding:**
   - Example text: "Hello, do you like tea? <|endoftext|> In someunknownPlace."
   - Encoding produces token IDs.
   - Decoding reconstructs the original text.
   - `<|endoftext|>` is a special token with ID 50256 in GPT-2’s vocabulary of 50,257 tokens.

5. **Handling Unknown Words:**
   - BPE splits unfamiliar words like someunknownPlace into subwords or characters.
   - This allows tokenizers to handle any text without needing a special "unknown word" token.

![Text Embedding Techniques](/llm09.png)

6. **How BPE Works:**
   - Starts with characters as tokens (e.g., “a,” “b”).
   - Merges frequent combinations of characters into subwords (e.g., “d” + “e” → “de”).
   - Repeats the process until a vocabulary size is reached, combining subwords into larger units.

7. **Exercise:**
   - Use BPE on unknown words like “Akwirw ier” to:
     - Tokenize and print the token IDs.
     - Decode individual IDs to see the subword splits.
     - Reconstruct the original input to verify BPE's behavior.

8. **Key Advantage:**
   - BPE tokenization ensures robust handling of any text, even words absent in the training data, by decomposing them into smaller, known units.
   - This approach allows LLMs to generalize better and efficiently handle unseen vocabulary.


# 2.6 Data sampling with a sliding window

# Start Generation Here
### Key Points: Generating Input-Target Pairs for LLM Training

1. **Goal:**
   - Create input–target pairs for training the LLM using a next-word prediction task.
   - Inputs are a block of tokens, and targets are the next token in sequence.

2. **Input-Target Pair Example:**
   - For context size = 4:
     - Input (x): [290, 4920, 2241, 287]
     - Target (y): [4920, 2241, 287, 257]

3. **Next-Word Prediction:**
   - Each input token predicts the token shifted by one position.

4. **Sliding Window Approach:**
   - A sliding window moves across the dataset to generate overlapping input-target pairs.
   - **Stride:** Controls how far the window shifts.
     - Stride = 1: Overlapping inputs.
     - Stride = context size: Non-overlapping inputs.

5. **Efficient Data Loading with PyTorch:**
   - A custom dataset class (`GPTDatasetV1`) generates input-target pairs:
     - Inputs: Sequences of token IDs.
     - Targets: Shifted token IDs (next-word predictions).
   - **DataLoader:** Fetches input-target pairs in batches.

6. **Batch Processing:**
   - Tensors are created for both inputs and targets.
   - Example for batch size = 8:
     - Inputs: 8 rows of token sequences (e.g., 4 tokens each).
     - Targets: Corresponding next-word tokens for each row.

7. **Avoiding Overfitting:**
   - Stride settings help balance between:
     - Overlapping batches (stride = small value).
     - Non-overlapping batches (stride = context size).

8. **Tradeoffs and Hyperparameters:**
   - Batch size: Small batches use less memory but result in noisier updates during training.
   - Experimenting with parameters like context size, stride, and batch size helps optimize training.

9. **Key Code Features:**
   - Custom PyTorch Dataset and DataLoader handle large text datasets efficiently.
   - Generates batches of inputs and targets that can be directly used for LLM training.
   - This method ensures the model gets sufficient training examples while efficiently managing data using PyTorch’s capabilities.
# End Generation Here
