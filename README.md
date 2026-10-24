# Eye Contact Correction Using Deep Learning

## About the Project
Eye Contact Correction is a deep learning project that aims to correct eye gaze so that a person appears to look toward the camera during video calls and online interviews.

## Technologies Used
- Python
- PyTorch
- NumPy
- OpenCV

## Dataset
A synthetic eye-image dataset is used for training the model.

- Total images: 129,285
- Image dimensions: 36 × 60 pixels
- Image type: Grayscale
- Gaze-angle and head-pose metadata are available.

### Dataset Files
- `dataset_x_img.npy` – Eye images
- `dataset_y_gaze_angles.npy` – Gaze angles
- `dataset_x_head_angles.npy` – Head-pose angles
- `dataset_names.npy` – Image identifiers

##Work Completed

Prepared and checked the synthetic dataset.

Created input-target image pairs for training.

Developed the ECC-Net-inspired model.

Tested the model and its image-warping process.

Trained the model.

## Model Architecture
The model takes a grayscale eye image and target gaze angles as input.

The vector field is used for image warping, while the brightness map adjusts image brightness.

## Current Status
The model has been trained, and initial evaluation has been performed. However, the corrected output still needs improvement. Further work is required to improve correction quality and validate the results.

## Future Work
- Improve the quality of corrected eye images.
- Further validate the warping and preprocessing pipeline.
- Evaluate the model on natural eye images.
- Integrate the model with a webcam for real-time eye-contact correction.

## Research Reference
*Eye Contact Correction using Deep Neural Networks*, WACV 2020.

This project is an ECC-Net-inspired implementation for learning and experimentation.
