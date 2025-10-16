# MNIST Handwritten Digit Inpainting with GANs

This project is a hands-on exploration into the world of Generative Adversarial Networks (GANs), specifically applied to the task of image inpainting on the MNIST dataset. The core idea is to train a U-Net based generator to intelligently fill in missing or corrupted parts of a handwritten digit, while a discriminator network pushes it to create realistic and coherent results.

This repository contains the final code from an iterative development process, the pre-trained model that achieved the best results, and a detailed analysis of the different configurations that were tested.

## 🛠️ Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 🚀 Installation and Usage

To run this project on your local machine, you can follow the steps below.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Dwarvess/mnist_gan.git](https://github.com/Dwarvess/mnist_gan.git)
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd train.py
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Test with the Pre-trained Model**
    The repository includes the pre-trained champion model (`generator_model5.pth`). To test it with your own masked image, use the `--image` flag in the terminal:
    ```bash
    python train.py --image path/to/your/image.png
    ```

5.  **Train the Model (Optional)**
    To train the model from scratch with the optimal parameters, set `CONTROL_MODEL_FILE = 0` inside the `train.py` script and then run:
    ```bash
    python train.py
    ```

## 📈 Development Process & Results

The development process involved several iterations to find the optimal balance between visual quality (sharpness) and semantic correctness (not misinterpreting digits). The table below summarizes the key models developed.

| Model Name | Configuration | Stre
