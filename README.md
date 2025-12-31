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


### Model Selection Guide

| Model | Parameters | VRAM | Training Time | Quality |
|-------|------------|------|---------------|---------|
| `mt5-small` | 300M | 8GB | Fast | Good |
| `mt5-base` | 580M | 16GB | Moderate | Better |
| `mt5-large` | 1.2B | 24GB+ | Slow | Best |

**Recommendation**: Start with `mt5-small` for experimentation, use `mt5-base` for final submission.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Deep Past Initiative and Kaggle for hosting the competition
- Google Research for the mT5 pre-trained models
- The Assyriology community for providing the training data

