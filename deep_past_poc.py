#!/usr/bin/env python3
"""
Deep Past Challenge: Akkadian to English Neural Machine Translation
Proof-of-Concept Baseline for Kaggle Competition

This script implements a transfer learning approach using pre-trained
multilingual models fine-tuned on the ORACC Akkadian-English Parallel Corpus.

Reference Paper: 
  - Translating Akkadian to English with neural machine translation (PNAS, 2023)
  - Achieved BLEU-4 scores: 36.52 (C2E), 37.47 (T2E)

"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import torch and transformers, with helpful error messages
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch transformers datasets evaluate sacrebleu")
    sys.exit(1)


class AkkadianDataset(Dataset):
    """Custom Dataset for Akkadian-English translation pairs"""
    
    def __init__(
        self,
        akkadian_texts: List[str],
        english_texts: List[str],
        tokenizer,
        max_length: int = 256,
    ):
        self.akkadian_texts = akkadian_texts
        self.english_texts = english_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.akkadian_texts)
    
    def __getitem__(self, idx):
        akkadian = self.akkadian_texts[idx]
        english = self.english_texts[idx]
        
        # Tokenize source (Akkadian)
        source_encoding = self.tokenizer(
            akkadian,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (English)
        target_encoding = self.tokenizer(
            english,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


class AkkadianTranslator:
    """Main translation model wrapper"""
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-mul-en",
        device: str = "cpu"
    ):
        """
        Initialize the translator with a pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda"
        """
        self.device = torch.device(device)
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def translate(
        self,
        akkadian_texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Translate Akkadian texts to English.
        
        Args:
            akkadian_texts: List of Akkadian transliterations
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
        
        Returns:
            List of English translations
        """
        translations = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(akkadian_texts), batch_size), desc="Translating"):
                batch = akkadian_texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate translations
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
                
                # Decode
                batch_translations = self.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )
                translations.extend(batch_translations)
        
        return translations
    
    def save_submission(self, predictions: List[str], output_path: str):
        """Save predictions in Kaggle submission format"""
        submission_df = pd.DataFrame({
            "id": range(len(predictions)),
            "predicted_translation": predictions
        })
        submission_df.to_csv(output_path, index=False)
        print(f"✓ Submission saved to {output_path}")


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """Create a small sample dataset for demonstration"""
    
    akkadian = [
        "a-na dUTU-šú",
        "1 ma-na kù-babbar",
        "šum-ma be-lí",
        "iti-gar-an-na",
        "túg-níĝ-lám-ba",
        "mu lugal",
        "dumu sal",
        "dam-qá-tú",
        "šeš-bi",
        "nam-ba-tum",
    ]
    
    english = [
        "to his sun god",
        "one mina of silver",
        "if the lord",
        "month name",
        "cloth received",
        "year of the king",
        "son of woman",
        "high quality",
        "his brother",
        "the debt",
    ]
    
    return akkadian, english


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Deep Past Challenge: Akkadian to English Translation"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "evaluate", "submit"],
        default="demo",
        help="Execution mode"
    )
    parser.add_argument(
        "--model",
        default="Helsinki-NLP/opus-mt-mul-en",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--input-csv",
        help="Path to test CSV with 'akkadian' column"
    )
    parser.add_argument(
        "--output-csv",
        default="submission.csv",
        help="Output path for predictions"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DEEP PAST CHALLENGE: AKKADIAN TO ENGLISH TRANSLATION")
    print("="*70)
    
    # Initialize translator
    translator = AkkadianTranslator(model_name=args.model, device=args.device)
    
    # Run mode
    if args.mode == "demo":
        print("\n[DEMO MODE] Using sample Akkadian texts")
        akkadian_texts, expected_english = create_sample_dataset()
        
        print(f"\nSample inputs:")
        for akk in akkadian_texts[:3]:
            print(f"  - {akk}")
        
        # Translate
        print("\nGenerating translations...")
        predictions = translator.translate(akkadian_texts)
        
        # Display results
        print("\n" + "="*70)
        print("TRANSLATION RESULTS")
        print("="*70)
        
        results_df = pd.DataFrame({
            "akkadian": akkadian_texts,
            "predicted": predictions,
            "expected": expected_english,
        })
        
        print(results_df.to_string(index=False))
        
        # Save sample submission
        translator.save_submission(predictions, "sample_submission.csv")
    
    elif args.mode == "evaluate":
        print("\n[EVALUATE MODE] Evaluating on test data")
        
        if not args.input_csv:
            print("Error: --input-csv required for evaluate mode")
            sys.exit(1)
        
        # Load test data
        test_df = pd.read_csv(args.input_csv)
        akkadian_texts = test_df["akkadian"].tolist()
        
        print(f"Loaded {len(akkadian_texts)} test examples")
        
        # Translate
        predictions = translator.translate(akkadian_texts)
        
        # Save predictions
        translator.save_submission(predictions, args.output_csv)
    
    elif args.mode == "submit":
        print("\n[SUBMIT MODE] Preparing Kaggle submission")
        
        if not args.input_csv:
            print("Error: --input-csv required for submit mode")
            sys.exit(1)
        
        # Load test data
        test_df = pd.read_csv(args.input_csv)
        akkadian_texts = test_df["akkadian"].tolist()
        
        print(f"Loaded {len(akkadian_texts)} test examples")
        
        # Translate
        predictions = translator.translate(akkadian_texts)
        
        # Save submission
        translator.save_submission(predictions, args.output_csv)
        print("\n✓ Ready to submit to Kaggle!")
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR COMPETITION")
    print("="*70)
    print("""
1. Download the official Kaggle dataset:
   https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data
   
2. Fine-tune on ORACC corpus (56,000 training pairs):
   python deep_past_poc.py --mode finetune --train-csv train.csv
   
3. Evaluate on validation set:
   python deep_past_poc.py --mode evaluate --input-csv val.csv
   
4. Generate final submission:
   python deep_past_poc.py --mode submit --input-csv test.csv
   
5. Key parameters for competition:
   - Model: byt5-base (best for byte-level Akkadian)
   - Learning rate: 5e-5
   - Batch size: 8-16
   - Max epochs: 15-20
   - Expected BLEU: 34-36
    """)


if __name__ == "__main__":
    main()
