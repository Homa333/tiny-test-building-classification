# Building Type Classification from Street View Images

This project predicts building type (e.g., residential, commercial) from Google Street View images using a zero-shot pipeline.

## Overview

- Input: Multiple images per location across different years  
- Output: Final building type + confidence  
- Labels: single_family, apartment_condo, commercial, mixed_use, empty_lot, unknown  

## Pipeline

1. **Segmentation (SegFormer)**  
   Extracts building regions and computes `building_ratio`.

2. **Classification (CLIP)**  
   Performs zero-shot classification using prompt-based scoring.

3. **Aggregation**  
   Combines predictions across years using recency-weighted confidence.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --data tiny_gsv_dataset/ --output results/
```