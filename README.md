

# Deep Past Initiative: Automated Translation and Restoration of Akkadian Cuneiform

## 1. Project Overview

This repository contains the source code and documentation for the **Deep Past Initiative Machine Translation Challenge**. The objective of this project is to develop a Neural Machine Translation (NMT) system capable of translating transliterated Old Assyrian cuneiform into English.

Given the low-resource nature of Akkadian (approximately 1,500 aligned sentence pairs in the training set), this solution leverages **Transfer Learning** using the **Multilingual T5 (mT5)** architecture. The pipeline implements a two-stage training strategy: domain adaptation via Masked Language Modeling (MLM) on unaligned scholarly publications, followed by supervised fine-tuning on the parallel corpus.

## 2. Methodology

### 2.1 Model Architecture

We utilize **mT5 (Multilingual Text-to-Text Transfer Transformer)**, a pre-trained encoder-decoder model covering 101 languages. This architecture is selected for its ability to transfer universal linguistic knowledge to low-resource languages and its flexibility in handling both translation and text restoration tasks within a unified text-to-text framework.

### 2.2 Training Pipeline

1. **Domain Adaptation (Pre-training):**
The model is first trained on the `publications.csv` dataset using a **Span Corruption** objective (Masked Language Modeling). Random spans of the Akkadian text are masked with sentinel tokens (e.g., `<extra_id_0>`), and the model learns to reconstruct the missing content. This step adapts the model's embeddings to Akkadian morphology and syntax prior to learning translation.


2. **Supervised Fine-Tuning:**
The domain-adapted model is then fine-tuned on `train.csv`. The task is framed as a sequence-to-sequence generation problem where the source is the Akkadian transliteration and the target is the English translation.

### 2.3 Evaluation Metric

Model performance is evaluated using the **Geometric Mean** of two metrics, as defined by the competition rules :

* **BLEU-4:** Measures n-gram overlap between the candidate and reference translations.


* **chrF++:** A character-level F-score that correlates well with human judgment for morphologically complex languages.



## 3. Repository Structure

deep-past-akkadian/
├── data/
│   ├── raw/
│   │   ├── train.csv           # Aligned training pairs (Akkadian -> English)
│   │   ├── test.csv            # Competition test set
│   │   └── publications.csv    # Unlabeled text for Domain Adaptation
│   └── processed/              # TFRecord files or cached tokenized datasets
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA of token distributions and vocabulary
│   └── 02_error_analysis.ipynb         # Qualitative analysis of model predictions
├── src/
│   ├── **init**.py
│   ├── config.py               # Configuration for hyperparameters (LR, Batch Size)
│   ├── data_loader.py          # tf.data pipelines for Span Corruption and Translation
│   ├── model.py                # Wrapper for TFAutoModelForSeq2SeqLM
│   ├── metrics.py              # Custom Callback for GeoMean calculation
│   └── utils.py                # Text normalization and cleaning utilities
├── train.py                    # Main entry point for training
├── predict.py                  # Inference script for generating submission files
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

## 4. Installation

### Prerequisites

* Python 3.8 or higher
* NVIDIA GPU (Recommended: 16GB VRAM minimum for batch size 16)
* TensorFlow 2.10+

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/deep-past-akkadian.git
cd deep-past-akkadian
pip install -r requirements.txt

```

**Key Dependencies:**

* `tensorflow`
* `transformers`
* `datasets`
* `sacrebleu`
* `sentencepiece`

## 5. Usage

### Configuration

Modify `src/config.py` to adjust hyperparameters such as learning rate, maximum sequence length, and model checkpoints.

* **Default Model:** `google/mt5-small` (for efficiency) or `google/mt5-base` (for performance).


* **Max Token Length:** 128 (Optimized for Old Assyrian sentence lengths).



### Training

Execute the training pipeline. You can choose to run the full pipeline or only fine-tuning.

```bash
# Run Domain Adaptation followed by Fine-Tuning
python train.py --mode full --epochs_pt 5 --epochs_ft 20

# Run Fine-Tuning only (using base mT5 weights)
python train.py --mode finetune --epochs_ft 20

```

### Inference

Generate the `submission.csv` file for the test set.

```bash
python predict.py \
    --model_dir./saved_models/akkadian_mt5_best \
    --input_file data/raw/test.csv \
    --output_file submission.csv

```

## 6. References

1. **Akkademia:** Gutherz, G., et al. (2023). "Translating Akkadian to English with neural machine translation." *PNAS Nexus*. 


2. **mT5:** Xue, L., et al. (2021). "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer."
3. **Masked Language Modeling for Ancient Languages:** Lazar, et al. (2021). "Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach." 


4. **Deep Past Initiative:** Kaggle Competition Overview.
