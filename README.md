# CleanVision using DIV2K : *Deep Learning based Real-Time Denoising, Inpainting, and Restoration of Noisy Images (Computer Vision)*

## 🧠 Project Overview

*CleanVision* is a deep learning-powered computer vision project designed to enhance, denoise, and restore degraded images in real time. Leveraging the *DIV2K* dataset, this project focuses on solving three core image restoration challenges: *denoising, **inpainting, and **super-resolution/restoration* using modern neural network architectures.

## 🌟 Key Features

- 🧹 Real-time image *denoising* using CNNs or GAN-based models  
- 🖼️ Intelligent *inpainting* of missing or corrupted image regions  
- 🎨 *Restoration* and super-resolution to enhance visual quality  
- 📦 Modular and extensible architecture for training and inference  
- 📊 Performance evaluation using PSNR, SSIM, and visual comparisons  
- 🧪 Support for custom inputs and real-world test cases  

## 📂 Dataset: DIV2K

We use the *DIV2K (DIVerse 2K Resolution)* dataset, a high-quality dataset for image restoration tasks. It contains 2K resolution images with various degradations, making it ideal for:

- Denoising tasks  
- Super-resolution  
- Compression artifact removal  
- Inpainting (custom masks applied)

*Source:* [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

## 🏗️ Model Architectures

- *Denoising*: DnCNN, UNet, or Residual Denoisers  
- *Inpainting*: Partial Convolutional Networks / Masked Autoencoders  
- *Restoration*: SRCNN / ESRGAN for super-resolution  
- *Loss Functions*: MSE, Perceptual Loss, SSIM Loss  
- *Optimizers*: Adam, AdamW with learning rate schedulers

> Optionally supports PyTorch Lightning or Keras for training workflows.

## 🧪 Workflow

1. *Data Preprocessing*  
   - Load and augment DIV2K images  
   - Apply synthetic noise, blur, or masking for training  
2. *Model Training*  
   - Train denoising, inpainting, and restoration models separately  
   - Use early stopping, checkpointing, and logging (TensorBoard/W&B)  
3. *Inference & Visualization*  
   - Pass noisy/damaged images  
   - Visualize original, corrupted, and restored results side by side  

## 🖼️ Example Use Case

| Input (Noisy) | Cleaned Output |
|---------------|----------------|
| ![noisy](sample_images/noisy.png) | ![clean](sample_images/clean.png) |

## 🔧 Installation & Setup

1. *Clone the repo*
   bash
   git clone https://github.com/imtapojit/CleanVision_using_DIV2K
   cd CleanVision-DIV2K
   

2. *Install dependencies*
   bash
   pip install -r requirements.txt
   

3. *Download and extract DIV2K*
   Place it in the data/ directory or update the path in the config.

4. *Train a model*
   bash
   python train.py --task denoising --model dncnn
   

5. *Run inference*
   bash
   python infer.py --input sample_images/noisy.png --task denoising
   

## 📊 Evaluation Metrics

- *PSNR* (Peak Signal-to-Noise Ratio)  
- *SSIM* (Structural Similarity Index)  
- *LPIPS* (Used for perceptual quality)

## 📌 Folder Structure


CleanVision-DIV2K/
├── data/                  # DIV2K and sample noisy images
├── models/                # Model architectures
├── outputs/               # Saved models, results
├── scripts/               # Utilities, evaluation
├── train.py               # Training script
├── infer.py               # Inference script
├── config.yaml            # Configurations
└── README.md              # Documentation


## 🚀 Future Work

- Add support for real-time webcam/image feed restoration  
- Integrate transformer-based vision models (e.g., ViT, Masked Autoencoders)  
- Train on additional datasets (e.g., BSD500, ImageNet subsets)  
- Add noise/adversarial attack robustness evaluation

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests for improvements.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- DIV2K Dataset by CVL ETH Zurich  
- PyTorch / TensorFlow open-source libraries  
- Research inspiration from ESRGAN, DnCNN, and NVIDIA’s inpainting models
