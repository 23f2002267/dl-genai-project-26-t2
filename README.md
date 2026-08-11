# Smart MCQ Solver Challenge: Deep Learning & GenAI Project Report

## 1. Executive Summary

This project addresses the **Smart MCQ Solver Challenge**, a multiple-choice question-answering task evaluated using the **Mean Average Precision at 3 (MAP@3)** metric. To systematically explore different modeling paradigms, three distinct architectures were designed, implemented, and evaluated:

1. **Model 1 (Pretrained Transformer):** Fine-tuned `roberta-base` for sequence classification, achieving a validation accuracy of **98.50%** and a **MAP@3 of 0.9921**.


2. **Model 2 (Custom Deep Learning):** A 3-layer Feedforward Neural Network (MLP) built from scratch over concatenated TF-IDF representations of prompts and options, achieving a validation accuracy of **93.50%** and a **MAP@3 of 0.9362**.


3. **Model 3 (Advanced GenAI RAG + PEFT):** A two-stage Dense Retrieve-and-Rank pipeline (Sentence-Transformer + Cross-Encoder) paired with Parameter-Efficient Fine-Tuning (LoRA) on `microsoft/deberta-v3-base`, achieving a validation accuracy of **51.75%** and a **MAP@3 of 0.6633**.



The `roberta-base` model was deployed as an interactive inference web application on a Hugging Face Space using Docker and Gradio. All experiments were logged using Weights & Biases (`wandb`).

---

## 2. Introduction

### Problem Statement

The objective of the competition is to select the correct answer out of 5 options (`A`, `B`, `C`, `D`, `E`) for technical domain-specific questions. Models are evaluated on their ability to rank candidate choices accurately, where predictions are submitted as top-3 ranked space-separated letters (e.g., `A B C`).

### Project Objective

* Prevent data leakage by enforcing group-aware validation splits via `GroupKFold`.


* Build a custom deep neural network from scratch to set a non-transformer baseline.


* Fine-tune transformer architectures (`roberta-base`) and optimize hyperparameters.


* Develop an advanced GenAI pipeline incorporating Retrieve-and-Rank RAG and Parameter-Efficient Fine-Tuning (PEFT / LoRA).


* Deploy the top-performing model to Hugging Face Spaces for interactive inference.

---

## 3. Dataset & Preprocessing

### Dataset Overview

* **Training Set:** 2,000 multiple-choice questions.


* **Test Set:** 500 questions.


* **Columns:** `id`, `prompt`, `A`, `B`, `C`, `D`, `E`, and `answer` (target label).



### Exploratory Data Analysis (EDA) Insights

* **Target Distribution:** Nearly balanced across option choices (`B`: 24.5%, `C`: 22.95%, `A`: 18.45%, `D`: 17.90%, `E`: 16.20%).


* **Length Statistics:**
* Average Prompt Length: ~18.15 words (max: 51 words).


* Average Option Length: ~26.0–26.5 words across options `A` through `E`.


* Average Combined Sequence Length: ~178.2 words (99th percentile: 490 words).




* **Domain & Syntax Patterns:** 8.35% contained code snippets (`def`, `class`, `import`), and 22.85% contained mathematical comparison/operator symbols.


* **Duplicate Prompts:** 242 duplicate prompts (12.10%) were identified. To avoid data leakage across train/validation splits, `GroupKFold(n_splits=5)` grouped by `prompt` was applied.



---

## 4. Tokenization Strategy

### Primary Tokenizer Selection (`roberta-base`)

* **Tokenizer Type:** Byte-Pair Encoding (BPE) fast tokenizer (`AutoTokenizer.from_pretrained("roberta-base")`).


* **Justification:** BPE handles domain-specific technical vocabulary, code syntax, and math symbols effectively by breaking unknown words into subword units.


* **Implementation:**
Each question prompt and option choice were formatted as pair sequences:

$$\text{First Sentence} = \text{"Question: "} + \text{prompt}$$


$$\text{Second Sentence} = \text{"Option "} + \text{col} + \text{": "} + \text{option\_text}$$



Input choices were tokenized with a maximum sequence length of `MAX_LEN = 256` and padded dynamically per batch via a custom multiple-choice data collator.



---

## 5. Modeling & Experimentation

### 5.1. Model 1: Fine-Tuned Pretrained Transformer (`roberta-base`)

* **Architecture:** `RobertaForMultipleChoice` based on the RoBERTa model (Liu et al., 2019).
#### Architectural Diagram 

```
             [ Question Prompt ]                     [ Choice Option (A-E) ]
                      │                                         │
                      ▼                                         ▼
            "Question: {prompt}"                    "Option {X}: {option_text}"
                      │                                         │
                      └───────────────────┬─────────────────────┘
                                          │
                                          ▼
                             [ BPE Tokenizer (RoBERTa) ]
                                 (Max Length = 256)
                                          │
                                          ▼
                       [ Input Pair Sequence Formatting ]
                  <s> Question: ... </s></s> Option X: ... </s>
                                          │
                                          ▼
                      [ RoBERTa-base Transformer Backbone ]
                         (12 Layers, 768 Hidden Dims,
                           12 Attention Heads)
                                          │
                                          ▼
                      [ Pooler / <s> Token Vector Representation ]
                                   (768 dims)
                                          │
                                          ▼
                         [ Linear Classification Head ]
                                   (768 -> 1)
                                          │
                                          ▼
                        [ Reshape to (Batch, 5) Logits ]
                                          │
                                          ▼
                        [ CrossEntropy Loss / MAP@3 Ranking ]

```

* **Salient Points:** Uses full self-attention mechanisms with dynamic masking over subword pairs to evaluate query-option compatibility.


* **Fine-Tuning Strategy:**
  * **Optimizer & Scheduler:** AdamW with linear warmup (100 warmup steps).


  * **Learning Rate:** $1.5 \times 10^{-5}$.


  * **Batch Size:** Per-device batch size = 4 with gradient accumulation steps = 2 (Effective batch size = 8).


  * **Regularization:** Label smoothing factor = 0.05, Weight decay = 0.01.


  * **Epochs:** 4 epochs with FP16 mixed precision enabled.





---

### 5.2. Model 2: Custom Deep Learning Model (Deep MLP from Scratch)

* **Architecture:** A 3-Layer Feedforward Neural Network with Batch Normalization and Dropout.



#### Architectural Diagram

```
              [ Prompt Text ]                 [ Option Text (A-E) ]
                     │                                  │
                     ▼                                  ▼
           [ TF-IDF Vectorizer ]              [ TF-IDF Vectorizer ]
              (5000 features)                    (5000 features)
                     │                                  │
                     └───────────────┬──────────────────┘
                                     ▼
                        [ Concatenation Layer ]
                             (10,000 dims)
                                     │
                                     ▼
                        [ Linear Layer: 10,000 -> 256 ]
                        [ BatchNorm1d + ReLU + Dropout(0.3) ]
                                     │
                                     ▼
                        [ Linear Layer: 256 -> 64 ]
                        [ BatchNorm1d + ReLU + Dropout(0.2) ]
                                     │
                                     ▼
                        [ Linear Layer: 64 -> 1 Score ]
                                     │
                                     ▼
                        [ Reshape to (Batch, 5) Logits ]

```

* **Salient Points:**
  * Extracts subword TF-IDF representations (5,000 max features) for all prompts and options.


  * Option candidate representations are concatenated with the prompt representation into a 10,000-dimensional vector.


  * Passed through dense scoring layers to output a single scalar logit per choice.





---

### 5.3. Model 3: Advanced GenAI Pipeline (Two-Stage Dense RAG + PEFT/LoRA)

* **Architecture:** RAG Retrieve-and-Rank pipeline combined with Low-Rank Adaptation (LoRA) on `microsoft/deberta-v3-base`.



#### Architectural Diagram

```
 [ User Prompt ] ────► [ Stage 1: Bi-Encoder Retriever ] ──► Top-15 Candidates
                        (all-MiniLM-L6-v2 Embeddings)
                                     │
                                     ▼
                       [ Stage 2: Cross-Encoder Reranker ] ──► Top-1 Reranked Context
                        (ms-marco-MiniLM-L-6-v2)
                                     │
                                     ▼
                       [ Context + Question Prompt + Option ]
                                     │
                                     ▼
                       [ DeBERTa-v3 Base Transformer ]
                                     │
                        ┌────────────┴────────────┐
                        │ LoRA Adapters (r=16)    │
                        │ Target: query_proj,     │
                        │         value_proj      │
                        └────────────┬────────────┘
                                     ▼
                       [ Multiple Choice Logits (A-E) ]

```

* **Salient Points:**
  * **Stage 1 (Bi-Encoder Retrieval):** Dense vector index over knowledge candidates using `sentence-transformers/all-MiniLM-L6-v2`.


  * **Stage 2 (Cross-Encoder Reranking):** Scores candidate relevance using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top context passage.


* **LoRA Configuration:**
  * Task Type: `SEQ_CLS`

  * Rank ($r$): 16, $\alpha$: 32, Dropout: 0.1


  * Target Modules: `query_proj`, `value_proj`

  * Trainable Parameters: **590,593 / 185,013,506 (0.3192%)**






---

## 6. Performance & Comparative Analysis

### Evaluation Metrics

* **Accuracy:** Top-1 correct choice percentage.


* **MAP@3 (Mean Average Precision at 3):** Official metric calculating position-weighted scores ($1.0$ for rank 1, $0.5$ for rank 2, $1/3$ for rank 3, $0.0$ otherwise).



### Summary Results Table

| Model Architecture | Strategy / Highlights | Trainable Params | Val Accuracy | Val MAP@3 |
| --- | --- | --- | --- | --- |
| **Model 1: RoBERTa-base** | Fine-Tuned Transformer Classifier | ~125M (100%) | **98.50%**<br> | **0.9921**<br> |
| **Model 2: Deep MLP** | Scratch Neural Net + TF-IDF Concatenation | ~2.5M (100%) | 93.50% | 0.9362 |
| **Model 3: DeBERTa-v3 + RAG** | Retrieve-and-Rank RAG + LoRA adapters | **590K (0.32%)**<br> | 51.75% | 0.6633 |

---

## 7. Conclusion & Future Work

* **Key Learnings:**
  * Full fine-tuning on high-capacity pretrained encoder models (`roberta-base`) delivers state-of-the-art sentence classification performance.
  * RAG + Rerank setup.

  * LoRA reduces trainable parameter count significantly (0.32% of base model size), enabling lightweight adaptation.




* **Challenges Overcome:**
  * Handled data duplication via `GroupKFold` split strategies.

  * Handled uneven training using parameter tuning (`learning_rate`, `label_smoothing_factor`, `warmup_step`,`epoch`)

  * Mitigated FP16 unscaling gradient instability in DeBERTa-v3 by using Bfloat16 (`bf16=True`) precision.




* **Future Work:**
  * Scale RAG.
  * Experiment with higher LoRA ranks ($r=32, 64$) targeting all linear projection layers (`key_proj`, `dense`).



---

## 8. References

1. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
2. He, P., et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. arXiv:2006.03654.
3. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.

---

Everything is fully aligned with your notebook runs. You can now use this directly as your final submission document!
