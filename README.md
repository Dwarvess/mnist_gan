## About This Project

This project is a hands-on exploration into the world of Generative Adversarial Networks (GANs), specifically applied to the task of image inpainting on the MNIST dataset.
It was developed as a preparatory study for my upcoming graduation project, allowing me to build a solid foundation in deep learning techniques for image restoration. 
The core idea is to train a U-Net based generator to intelligently fill in missing or corrupted parts of an image, while a discriminator network pushes it to create
realistic and coherent results. This repository serves as both a learning exercise and a practical implementation of modern computer vision concepts.Below, I will 
guide you through the necessary steps to set up the environment, train the model, and test it with your own images.
## 🛠️ Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## ⚙️ Configuration Parameters
This script includes several parameters that can be adjusted to change the training behavior. Here's a brief explanation of each:

* **`TRAIN_MASK_MODE`**: Determines the type of mask applied to images during training. "mixed" is recommended for robustness.
* **`BATCH_SIZE`**: Defines the number of samples processed in a single iteration.
* **`NUM_EPOCHS`**: Specifies the total number of training passes over the entire dataset.
* **`LR` (Learning Rate)**: Controls the step size of the optimizer.
* **`LAMBDA_RECON`**: A crucial coefficient that balances the GAN Loss (realism) and the Reconstruction Loss (faithfulness).
* **`CONTROL_MODEL_FILE`**: A flag to either load a pre-trained model (1) or train from scratch (0).
