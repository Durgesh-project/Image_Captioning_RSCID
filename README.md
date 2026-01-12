# Remote Sensing Image Captioning using CNN–Transformer

This repository presents an end-to-end deep learning system for automatic caption generation of remote sensing images using a CNN–Transformer architecture. The project is implemented in the notebook `Image_Captioning_RSICD_ResNet50_Transformer.ipynb` and is trained and evaluated on the RSICD (Remote Sensing Image Caption Dataset).

The model combines a ResNet50 convolutional encoder for visual feature extraction with a Transformer-based decoder employing multi-head attention for natural language sequence generation. The system is designed to handle complex aerial scenes and generate semantically accurate and grammatically coherent descriptions.

---

## Dataset

The project uses the **RSICD (Remote Sensing Image Caption Dataset)**.

Source:  
https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset  

Provider: Arto (Hugging Face)  
License: CC0 1.0 Universal (Public Domain)

### Dataset Structure

The dataset is provided in three CSV files:

- `train.csv` – Training split  
- `valid.csv` – Validation split  
- `test.csv` – Test split  

Each file contains:

| Column   | Description |
|----------|-------------|
| filename | Image file name |
| captions | Multiple human-annotated captions for each image |

Each image is paired with several captions, enabling robust learning of visual–language correspondences.

---

## Model Architecture

### Encoder: ResNet50
- Pretrained on ImageNet  
- Extracts high-level spatial feature maps  
- Features are reshaped into a sequence of visual tokens  

### Decoder: Transformer
- Multi-head self-attention for language modeling  
- Cross-attention between image features and word embeddings  
- Positional encodings  
- Feed-forward layers and residual connections  
- Autoregressive caption generation  

---

## Method Summary

A ResNet50 encoder extracts spatial feature maps from remote sensing images. These features are projected into a Transformer decoder, which uses multi-head cross-attention to align visual regions with linguistic tokens. Captions are generated autoregressively using beam search decoding, enabling accurate and fluent descriptions of complex aerial scenes.

---

## Training and Optimization Techniques

- GloVe embeddings for semantic word initialization  
- Teacher forcing for stable sequence learning  
- Label smoothing for regularization  
- Gradient clipping to prevent exploding gradients  
- Learning rate scheduling for convergence  
- Beam search for optimal decoding  
- Attention map visualization for interpretability  

---

## Results

Based on the outputs in `Image_Captioning_RSICD_ResNet50_Transformer.ipynb`, the model achieves:

- **Best Validation BLEU-4:** 0.4258  
- **Best Test BLEU-4 (Beam Search):** 0.2742  

These results demonstrate effective caption generation for complex aerial scenes using multi-head attention and Transformer-based decoding.

---

## Repository Contents

- `Image_Captioning_RSICD_ResNet50_Transformer.ipynb`  
  - Data preprocessing  
  - Vocabulary construction  
  - GloVe embedding integration  
  - CNN–Transformer implementation  
  - Training and evaluation  
  - Beam search inference  
  - BLEU score computation  
  - Attention visualization  

---

## Installation

```bash
pip install tensorflow torch torchvision numpy pandas nltk matplotlib pillow
```

Download the RSICD dataset and place the images and CSV files according to the paths used in the notebook.

---

## Usage

```bash
jupyter notebook Image_Captioning_RSICD_ResNet50_Transformer.ipynb
```

The notebook performs:
1. Dataset loading and preprocessing  
2. Model training  
3. Validation and testing  
4. Caption generation  
5. Attention map visualization  

---

## Applications

- Automatic annotation of satellite imagery  
- Remote sensing scene understanding  
- Vision–language research  
- Geospatial information retrieval  
- Earth observation systems  

---

## Acknowledgements

This work uses the **RSICD (Remote Sensing Image Caption Dataset)**.  
We acknowledge the original RSICD authors and **Arto (Hugging Face)** for providing and distributing the dataset on Kaggle.

If used in academic work, please cite the original RSICD dataset publication.

---

## Author

Durgesh  
Indian Institute of Technology Bombay (IIT Bombay)  
Environmental Science and Engineering Department (ESED)

---

## License

This project is released under the MIT License.
