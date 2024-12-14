# Chapter 1: Understanding Large Language Models


&nbsp;
## Main Chapter Code

There is no code in this chapter.


&nbsp;
## Bonus Materials

As optional bonus material, below is a video tutorial where I explain the LLM development lifecycle covered in this book:

<br>
<br>




[![Link to the video](https://img.youtube.com/vi/kPGTx4wcm_w/0.jpg)](https://www.youtube.com/watch?v=kPGTx4wcm_w)




# 1.1 What is an LLM
### Definition and Purpose:
- A Large Language Model (LLM) is a neural network designed to understand, generate, and respond to human-like text.
- It is trained on massive datasets, often including much of the publicly available internet text.

### Why “Large”?
- Refers to both the size of the model (tens or hundreds of billions of parameters) and the scale of the training data.
- Parameters are the adjustable weights optimized during training, enabling the model to predict the next word in a sequence.

### Key Technique:
- LLMs use a transformer architecture, which selectively focuses on relevant parts of the input text to make predictions.
- This ability makes them highly effective at understanding the context, nuances, and relationships within language.

### Generative AI:
- LLMs are a form of generative artificial intelligence (GenAI), capable of creating text, answering questions, and performing other language-based tasks.

### Connection to Deep Learning:
- LLMs are a specific application of deep learning, a branch of machine learning that uses multilayered neural networks to model complex patterns.
- Unlike traditional machine learning, deep learning does not require manual feature extraction, as the model learns patterns directly from the raw data.

### Applications and Challenges:
- LLMs solve various problems like text generation, classification, and translation.
- Future chapters will explore their architecture, training process (e.g., next-word prediction), and challenges in more detail.

### Comparison to Traditional Machine Learning:
- Traditional machine learning relies on manual feature extraction by experts, while deep learning automates this process through neural networks.
- Example: Spam filters—traditional models require predefined features, whereas deep learning models learn patterns from raw data.

![LLM Image](/llm.png)

# 1.3 Stages of building and using LLMs

# Start Generation Here
### Thought about building custom LLMs for a second

#### Why Build Your Own LLM?
- Understand how LLMs work and what their limits are by creating one from scratch.
- Make custom models that fit your specific needs better than general-purpose models like ChatGPT.

#### Privacy and Control
- Keep your sensitive data private by not sharing it with big third-party companies.
- Run smaller models directly on your devices (like a laptop or phone) to save on costs and reduce delays.
- Have full control over updates and changes to your model.

#### Two-Stage Training: Pretraining + Fine-Tuning
- **Pretraining:** First, train the model on a very large, varied dataset so it learns general language skills.
- **Fine-Tuning:** Next, train it on a smaller, more specific dataset to focus on your particular use case.

![Pretraining an LLM](/llm01.png)


# 1.4 Introducing the transformer architecture

## Main Idea:
The paper “Attention Is All You Need” introduces the Transformer, a new type of model for processing language and other sequence data. Instead of using traditional methods that rely on sequences processed step-by-step (like recurrent networks) or complex filters (like convolutional networks), the Transformer uses only “attention” to figure out which parts of the input are important. This makes it simpler, faster, and more efficient to train.

## Key Points:

### Background:
Older models for understanding and generating sentences, like machine translation systems, often used Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs). These can be slow, complicated, and hard to parallelize (run many steps at the same time).

### The Transformer’s Core Idea—Attention:
Instead of processing words one-by-one in order, the Transformer uses a method called attention. Attention allows the model to look at all words at once and decide which ones are most important to understand the meaning. This makes the model work more efficiently, since it’s not forced to go through the sentence word by word.

### No More Loops or Filters:
By removing the need for recurrence (loops) or convolution layers, the Transformer is simpler and can be trained more quickly. It can handle long sentences better because it can directly compare any two words, regardless of how far apart they are.

### Better Performance:
When tested on tasks like translating English to German or English to French, the Transformer model performed better than previous best models. It reached higher quality scores (measured by something called BLEU) and did so while training much faster.

### More Parallelization:
Because the Transformer doesn’t process sentences in a strict order, it can run parts of its work at the same time. This makes it much faster on modern computer hardware (like GPUs), greatly reducing training time.

### Generalizing Beyond Translation:
The paper also shows that the Transformer can be used for other tasks, such as understanding sentence structure (parsing). It works well even when there isn’t a lot of training data available, suggesting it’s a robust and flexible approach.

### In Simple Terms:
Think of the Transformer as a new recipe for teaching a computer to read and write languages. Instead of making the computer follow a complicated list of steps that look at words one-by-one, we give it a simple trick—attention—that helps it instantly see which words matter most. This trick makes the computer learn faster, run faster, and give better translations and understand language in a smarter way.


# Simple LLM Architecture
![Simple LLM Image](/Simple%20LLM%20Architecture.png)


# The transformer architecture consists of two submodules:

## Two-Part Structure (Encoder + Decoder)

The Transformer has two main parts: an encoder that reads the input text and a decoder that produces the output text. The encoder turns the input words into meaningful numerical patterns (vectors). The decoder takes these vectors and uses them to generate the final output, such as a translated sentence.

## Self-Attention

Both the encoder and decoder consist of many layers connected by a so-called self-attention mechanism

Both the encoder and decoder use a technique called self-attention. This lets the model figure out which words in the sentence matter most to understand the meaning, even if they are far apart. We’ll explain self-attention in detail later (in chapter 3).

## Influential Models (BERT and GPT)
* BERT (short for bidirectional encoder representations from transformers)
* GPT (generative pre-trained transformer)

Models like BERT and GPT are special versions of the Transformer. BERT focuses on understanding text (it predicts missing words and is good at classification tasks). GPT focuses on generating text (like writing a paragraph or continuing a story).

## Different Strengths

GPT models are great for creating new text based on what they’ve read. BERT models are great for understanding text and figuring out the right labels (like deciding if a tweet is toxic or not).

## Real-World Use

Companies like X (formerly Twitter) use BERT-based models to detect harmful or toxic language in content.

# Tranformer's encoder-decoder
![Image Description](/Tranformer's%20encoder-decoder.png)

## GPT Models Are Very Flexible:
Although GPT models are mainly built to finish sentences, they can do much more. They can handle tasks they weren’t directly taught to do.

## Zero-Shot Learning:
Zero-shot means the model can figure out how to do a new task without seeing any examples first. For instance, if you ask a GPT model to summarize a text, even if it was never directly trained to summarize, it can try to do it right away.

## Few-Shot Learning:
Few-shot means the model can learn from just a few examples. If you show it a couple of examples of how to solve a math problem, it can then try to solve a new one on its own, even though it hasn’t been trained specifically for that task.

## In Other Words:
GPT models are flexible enough to handle tasks they haven’t seen before (zero-shot) and can quickly pick up new tasks after seeing just a few examples (few-shot), making them very powerful tools.
# Additional Image

![GPT Flexibility](/zero%20shot%20vs%20few%20shot%20learning.png)

# Transformer vs LLM


# Transformers are the Blueprint for Many LLMs:
Most large language models (LLMs) today follow the design of the transformer architecture.

# Not All Transformers are LLMs:
While transformers are usually linked to language processing, they can also be used in other areas, like analyzing images. So, just because something is a transformer, it doesn’t mean it’s a large language model.

# Not All LLMs are Transformers:
Some large language models are built using older methods (like RNNs or CNNs) instead of transformers. These alternatives exist to try to make LLMs run faster and more efficiently.

# It’s Still an Open Question:
We don’t know yet if these non-transformer LLMs will become as good or popular as transformer-based LLMs like GPT. For now, transformer-based models dominate the field.

# For Simplicity:
In most discussions, people treat “LLM” and “transformer-based LLM” as the same, because transformer designs are currently the standard approach.

# 1.5 Utilizing large datasets
## Huge Amounts of Text:
Models like GPT and BERT are trained on enormous collections of text—billions of words. These words cover many subjects, from science and history to pop culture, and can include different human languages as well as programming languages.

## Why Such Big Datasets?
By learning from such a wide range of texts, the model gets a broad, general understanding of language and knowledge. This helps it handle many types of questions and topics.

## Concrete Example—GPT-3:
For GPT-3, which was the starting point for ChatGPT, the creators used a very large and varied dataset. Table 1.1 in the document shows the details of what texts were included, illustrating just how extensive and diverse the source material was.


## Dataset Details for GPT-3:
GPT-3 was trained on 300 billion tokens, even though the dataset collected had 499 billion tokens. The authors didn’t explain why they didn’t use all the data. For example, a major dataset in GPT-3, CommonCrawl, contains 410 billion tokens, taking up about 570 GB of storage.

## Comparison with Other Models:
Newer models, like LLaMA, use additional datasets such as Arxiv research papers (92 GB) and StackExchange Q&As (78 GB) for broader training.

## Public Alternatives to GPT-3 Data:
While GPT-3’s exact training data isn’t shared, a similar open dataset, Dolma, includes 3 trillion tokens. Note: Dolma may include copyrighted works, and its use depends on legal and ethical guidelines.

## Pretrained Models as Foundations:
Pretrained models like GPT-3 are called base models because they can be adapted (fine-tuned) for specific tasks. Fine-tuning is less resource-intensive than pretraining and uses smaller datasets to improve performance for specific applications.

## Cost of Pretraining:
Pretraining a model like GPT-3 is very expensive, costing around $4.6 million in cloud computing credits. This cost makes using existing pretrained models (open-source or proprietary) more practical.

## Using Pretrained Models for Learning:
You can skip the costly pretraining step by using existing open-source model weights (already trained data) and adapting them. This approach allows fine-tuning or experimentation on consumer hardware, making it accessible for educational purposes.

## What We Will Learn:
You’ll learn to implement pretraining code for an LLM. You’ll also practice loading existing pretrained models into this code, skipping the expensive pretraining stage and focusing on fine-tuning for specific tasks.


# 1.6 A closer look at the GPT architecture

## 1.6.1 What is GPT?
GPT stands for Generative Pretrained Transformer. It was first introduced by OpenAI in the paper “Improving Language Understanding by Generative Pre-Training.” GPT-3 is a much larger and improved version of the original model with:
- More parameters (175 billion in GPT-3).
- Training on larger datasets for better understanding of language.

## 1.6.2 How is GPT-3 Used in ChatGPT?
ChatGPT is based on GPT-3 but was fine-tuned using a special dataset of instructions and responses. This method is detailed in OpenAI’s InstructGPT paper. Fine-tuning GPT-3 this way allows ChatGPT to handle specific tasks like:
- Spelling correction.
- Classification.
- Language translation.
- Context-aware text generation.

## 1.6.3 Pretraining Task (Next-Word Prediction):
GPT learns by predicting the next word in a sentence. For example, given the phrase “The cat is on the,” the model predicts the next word is “mat.” This process uses self-supervised learning:
- No manual labels are required. The next word itself becomes the label during training.
- This allows training on massive unlabeled datasets efficiently.

## 1.6.4 GPT’s Architecture:
GPT is based on the transformer architecture but only uses the decoder part:
- Unlike the original transformer, it does not have an encoder.
- The model generates text one word at a time, using previous outputs to predict the next word. This is called an autoregressive model because it builds predictions based on past data.

## 1.6.5 Why is GPT-3 Powerful?
Larger Scale:
- The original transformer had 6 encoder-decoder layers.
- GPT-3 has 96 layers and 175 billion parameters for greater capacity. This scale allows GPT-3 to:
  - Understand and generate more complex language patterns.
  - Perform well on a variety of tasks without specific training.

### Key Takeaways:
- GPT models excel at generating coherent and contextually accurate text by predicting the next word in a sequence.
- Fine-tuning (as done for ChatGPT) makes GPT even more versatile for specific tasks.
- Its simplicity (using only the decoder) combined with its large scale makes GPT-3 a powerful model for language understanding and generation.

## Image of GPT-3
![GPT-3 Architecture](/gpt-3.png)




# 1.7 Building a large language model

## Overview:
The process of building an LLM like GPT involves three main stages:

1. **Stage 1: Foundation and Attention Mechanism**
   - **What you’ll learn:**
     - Basic data preprocessing steps needed for training a model.
     - How to code the attention mechanism, which is the core of all LLMs.
   - **Goal:**
     - Understand and implement the key building blocks of LLMs, focusing on how attention works to help models understand relationships between words.

2. **Stage 2: Pretraining the Model**
   - **What you’ll do:**
     - Code and pretrain a small, GPT-like model capable of generating text.
     - Learn how to evaluate the model’s performance, which is crucial for improving its capabilities.
   - **Key Points:**
     - Full-scale pretraining is expensive (costing thousands to millions of dollars).
     - This step focuses on educational implementation using a small dataset to demonstrate the concept.
     - Includes examples of how to load pretrained model weights from open-source LLMs to save time and resources.

3. **Stage 3: Fine-Tuning the Model**
   - **What you’ll do:**
     - Take a pretrained LLM and fine-tune it for specific tasks like:
       - Answering questions.
       - Classifying texts.
       - Following instructions.
   - **Why this is important:**
     - Fine-tuning adapts a general-purpose model to specific real-world applications, making it useful for tasks in research and industry.

## Conclusion:
### Exciting Journey Ahead:
By working through these stages, you’ll gain a hands-on understanding of building and refining LLMs.

### Final Outcome:
You’ll know how to create and adapt models for practical applications, including using open resources to optimize training costs.

It’s a step-by-step journey into the world of LLMs, starting with the basics and ending with practical, real-world applications!

![Building a Large Language Model](/build-llm.png)

## Summary
- LLMs have transformed the field of natural language processing, which previously mostly relied on explicit rule-based systems and simpler statistical methods.
- The advent of LLMs introduced new deep learning-driven approaches that led to advancements in understanding, generating, and translating human language.
- Modern LLMs are trained in two main steps:
  - First, they are pretrained on a large corpus of unlabeled text by using the prediction of the next word in a sentence as a label.
  - Then, they are fine-tuned on a smaller, labeled target dataset to follow instructions or perform classification tasks.
- LLMs are based on the transformer architecture. The key idea of the transformer architecture is an attention mechanism that gives the LLM selective access to the whole input sequence when generating the output one word at a time.
- The original transformer architecture consists of an encoder for parsing text and a decoder for generating text.
- LLMs for generating text and following instructions, such as GPT-3 and ChatGPT, only implement decoder modules, simplifying the architecture.
- Large datasets consisting of billions of words are essential for pretraining LLMs.
- While the general pretraining task for GPT-like models is to predict the next word in a sentence, these LLMs exhibit emergent properties, such as capabilities to classify, translate, or summarize texts.
- Once an LLM is pretrained, the resulting foundation model can be fine-tuned more efficiently for various downstream tasks.
- LLMs fine-tuned on custom datasets can outperform general LLMs on specific tasks.

