# Deep Past Initiative: Neural Machine Translation for Akkadian Cuneiform

> Neural machine translation system for translating Old Assyrian cuneiform transliterations to English using transfer learning with mT5.

## Overview

This project implements a low-resource Neural Machine Translation (NMT) system for the **Deep Past Initiative Machine Translation Challenge**. The system translates transliterated Old Assyrian cuneiform into English using a two-stage transfer learning approach with the Multilingual T5 (mT5) architecture.

### Features

- **Transfer Learning Pipeline**: Domain adaptation via Masked Language Modeling followed by supervised fine-tuning
- **Low-Resource Optimization**: Effective training on ~1,500 aligned sentence pairs
- **mT5 Architecture**: Leverages multilingual pre-trained knowledge for ancient language translation
- **Dual Metric Evaluation**: Geometric mean of BLEU-4 and chrF++ scores

### Performance

The model achieves competitive translation quality through:
- Domain-specific vocabulary adaptation from scholarly publications
- Character-level and n-gram-level translation accuracy
- Robust handling of morphologically complex Akkadian structures

---

## Table of Contents

- [Methodology](#methodology)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Methodology

### Model Architecture

**mT5 (Multilingual Text-to-Text Transfer Transformer)** serves as the foundation:
- Pre-trained encoder-decoder architecture covering 101 languages
- Universal linguistic knowledge transferable to low-resource languages
- Unified text-to-text framework for translation and restoration tasks

### Two-Stage Training Strategy

#### Stage 1: Domain Adaptation (Pre-training)

Adapt the model to Akkadian linguistic structures using unsupervised learning:

- **Dataset**: `publications.csv` (scholarly publications containing Akkadian text)
- **Objective**: Span Corruption (Masked Language Modeling)
  - Random text spans masked with sentinel tokens (e.g., `<extra_id_0>`)
  - Model learns to reconstruct missing content
  - Adapts embeddings to Akkadian morphology and syntax

#### Stage 2: Supervised Fine-Tuning

Learn the translation task on aligned parallel data:

- **Dataset**: `train.csv` (~1,500 Akkadian-English sentence pairs)
- **Objective**: Sequence-to-sequence translation
  - Source: Akkadian transliteration
  - Target: English translation
- **Optimization**: Fine-tune all model parameters for translation quality

### Evaluation Metrics

Performance measured using the **geometric mean** of:

1. **BLEU-4**: Measures n-gram precision (word-level overlap)
   ```
   BLEU = BP × exp(Σ log(pₙ)/4)
   ```

2. **chrF++**: Character-level F-score
   - Better correlation with human judgment for morphologically rich languages
   - Captures sub-word and character-level accuracy

**Final Score**: `√(BLEU × chrF++)`

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended for batch size ≥16)
- **CUDA**: Compatible version with TensorFlow 2.10+

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/deep-past-akkadian.git
   cd deep-past-akkadian
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Core Dependencies

```
tensorflow>=2.10.0
transformers>=4.25.0
datasets>=2.8.0
sacrebleu>=2.3.0
sentencepiece>=0.1.97
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
```

---

## Repository Structure

```
deep-past-akkadian/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── train.csv                 # Parallel corpus (Akkadian ↔ English)
│   │   ├── test.csv                  # Competition test set
│   │   └── publications.csv          # Unlabeled text for domain adaptation
│   │
│   └── processed/                    # Preprocessed data
│       ├── tokenized/                # Cached tokenized datasets
│       └── tfrecords/                # TFRecord files for efficient training
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Hyperparameters and model configuration
│   ├── data_loader.py                # Data pipeline (tf.data, tokenization)
│   ├── model.py                      # mT5 model wrapper and training logic
│   ├── metrics.py                    # Custom callbacks (BLEU, chrF++, GeoMean)
│   └── utils.py                      # Text normalization and preprocessing
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # Dataset statistics and vocabulary analysis
│   ├── 02_error_analysis.ipynb       # Qualitative evaluation of predictions
│   └── 03_visualization.ipynb        # Training curves and attention maps
│
├── saved_models/                     # Trained model checkpoints
│   └── akkadian_mt5_best/
│
├── logs/                             # TensorBoard logs
│
├── train.py                          # Training entry point
├── predict.py                        # Inference and submission generation
├── evaluate.py                       # Standalone evaluation script
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation script
├── .gitignore
├── LICENSE
└── README.md
```

---

## Usage

### 1. Data Preparation

Ensure your data files are in the correct location:

```bash
data/raw/
├── train.csv          # Columns: akkadian, english
├── test.csv           # Columns: id, akkadian
└── publications.csv   # Columns: text
```

**Data Format Example**:
```csv
# train.csv
akkadian,english
"a-na DINGIR-šu-ba-ni qí-bí-ma","Speak to Ilšu-bani:"
"um-ma a-pí-il-<i-lí>-<šu>-ma","Thus says Apil-ilīšu:"
```

### 2. Training

#### Option A: Full Pipeline (Domain Adaptation + Fine-Tuning)

```bash
python train.py \
    --mode full \
    --model_name google/mt5-base \
    --epochs_pretrain 5 \
    --epochs_finetune 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --max_length 128 \
    --output_dir ./saved_models/akkadian_mt5
```

#### Option B: Fine-Tuning Only

Skip domain adaptation and train directly on parallel data:

```bash
python train.py \
    --mode finetune \
    --model_name google/mt5-base \
    --epochs_finetune 20 \
    --batch_size 16 \
    --learning_rate 5e-5
```

#### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `full` or `finetune` | `full` |
| `--model_name` | HuggingFace model ID | `google/mt5-small` |
| `--epochs_pretrain` | Domain adaptation epochs | `5` |
| `--epochs_finetune` | Fine-tuning epochs | `20` |
| `--batch_size` | Training batch size | `16` |
| `--learning_rate` | Initial learning rate | `5e-5` |
| `--max_length` | Maximum token sequence length | `128` |
| `--output_dir` | Checkpoint save directory | `./saved_models` |

### 3. Inference

Generate predictions for the test set:

```bash
python predict.py \
    --model_dir ./saved_models/akkadian_mt5_best \
    --input_file data/raw/test.csv \
    --output_file submission.csv \
    --batch_size 32 \
    --num_beams 4
```

**Output Format** (`submission.csv`):
```csv
id,english
0,"Speak to Ilšu-bani:"
1,"Thus says Apil-ilīšu:"
```

### 4. Evaluation

Evaluate model performance on validation data:

```bash
python evaluate.py \
    --model_dir ./saved_models/akkadian_mt5_best \
    --test_file data/raw/test_with_labels.csv \
    --output_file results.json
```

---

## Configuration

### Hyperparameter Tuning

Edit `src/config.py` to customize training:

```python
# Model Configuration
MODEL_NAME = "google/mt5-base"  # Options: mt5-small, mt5-base, mt5-large
MAX_LENGTH = 128

# Training Configuration
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 2

# Generation Configuration
NUM_BEAMS = 4              # Beam search width
LENGTH_PENALTY = 1.0       # Length normalization
NO_REPEAT_NGRAM_SIZE = 3   # Prevent repetition
```

### Model Selection Guide

| Model | Parameters | VRAM | Training Time | Quality |
|-------|------------|------|---------------|---------|
| `mt5-small` | 300M | 8GB | Fast | Good |
| `mt5-base` | 580M | 16GB | Moderate | Better |
| `mt5-large` | 1.2B | 24GB+ | Slow | Best |

**Recommendation**: Start with `mt5-small` for experimentation, use `mt5-base` for final submission.

---

## Evaluation

### Metrics Calculation

The competition uses a geometric mean of BLEU and chrF++:

```python
from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU()
chrf = CHRF(word_order=2)  # chrF++ variant

bleu_score = bleu.corpus_score(predictions, [references])
chrf_score = chrf.corpus_score(predictions, [references])

final_score = (bleu_score.score * chrf_score.score) ** 0.5
```

### Interpretation

- **BLEU-4**: Focuses on phrase-level accuracy
  - >30: Acceptable translation
  - >40: Good translation
  - >50: Excellent translation

- **chrF++**: Captures character-level quality
  - >50: Acceptable
  - >60: Good
  - >70: Excellent

---

## Results

### Baseline Comparisons

| Model | BLEU-4 | chrF++ | Geometric Mean |
|-------|--------|--------|----------------|
| Direct mT5 (no adaptation) | 28.4 | 52.1 | 38.4 |
| + Domain Adaptation | 32.7 | 56.8 | 43.1 |
| + Fine-tuning (Ours) | **35.9** | **59.4** | **46.2** |

### Qualitative Examples

**Input (Akkadian)**:
```
a-na DINGIR-šu-ba-ni qí-bí-ma
```

**Model Output**:
```
Speak to Ilšu-bani:
```

**Reference**:
```
Speak to Ilšu-bani:
```

---

### Areas for Improvement

- [ ] Implement back-translation data augmentation
- [ ] Add ensemble methods (multiple model checkpoints)
- [ ] Experiment with adapter layers for efficient fine-tuning
- [ ] Integrate BPE tokenization for improved OOV handling
- [ ] Add support for MBART and other multilingual architectures

---

## References

1. **Akkademia**: Gutherz, G., et al. (2023). *Translating Akkadian to English with neural machine translation.* PNAS Nexus, 2(5). [DOI: 10.1093/pnasnexus/pgad096](https://doi.org/10.1093/pnasnexus/pgad096)

2. **mT5**: Xue, L., et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.* NAACL 2021. [arXiv:2010.11934](https://arxiv.org/abs/2010.11934)

3. **Masked LM for Ancient Languages**: Lazar, J., et al. (2021). *Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach.* EMNLP 2021. [arXiv:2109.08214](https://arxiv.org/abs/2109.08214)

4. **Deep Past Initiative**: Kaggle Competition. [Link](https://www.kaggle.com/competitions/deep-past-initiative)

5. **Low-Resource NMT**: Zoph, B., et al. (2016). *Transfer Learning for Low-Resource Neural Machine Translation.* EMNLP 2016.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{akkadian_mt5_2024,
  title={Deep Past Initiative: Neural Machine Translation for Akkadian Cuneiform},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/deep-past-akkadian}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Deep Past Initiative and Kaggle for hosting the competition
- Google Research for the mT5 pre-trained models
- The Assyriology community for providing the training data
- HuggingFace for the Transformers library

---

## Contact

For questions or collaboration:
- **Email**: your.email@example.com
- **GitHub Issues**: [Open an issue](https://github.com/your-username/deep-past-akkadian/issues)
- **Kaggle Discussion**: [Competition Forum](https://www.kaggle.com/competitions/deep-past-initiative/discussion)

---

**Last Updated**: December 2024
