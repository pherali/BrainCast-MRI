# Latent Diffusion Model for MRI Image Generation

This project implements a latent diffusion model in PyTorch to generate synthetic 2D brain MRI images.
The model is trained on the LGG MRI Segmentation dataset from Kaggle.

## Project Overview

The model uses a two-stage approach:
1.  Autoencoder: An autoencoder is first trained to compress the high-resolution MRI scans into a small, manageable latent space.
2.  U-Net Diffusion: A U-Net model is then trained to denoise this latent space. It learns to reverse a "forward diffusion" process, where noise is gradually added to the latent representations.

Image generation is achieved by starting with pure random noise and using the trained U-Net to iteratively denoise it into a clean latent summary, which is then upscaled by the autoencoder's decoder.

## How to Run This Project

1.  **Open the Notebook:** Click on the `.ipynb` file in this repository. To run it, click the "Open in Colab" badge at the top of the notebook.
2.  **Set Up Colab:** In the Colab notebook, make sure to select a GPU runtime (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3.  **Run the Code:** Run the cells in the notebook from top to bottom.

## Generating the Trained Models

The trained model files (`autoencoder.pth` and `unet.pth`) are not included in this repository due to their size.

The first time you run the entire notebook, the script will automatically train the models from scratch and save them to your Google Drive. On all subsequent runs, the script will detect these saved files and load them directly, skipping the lengthy training process.

