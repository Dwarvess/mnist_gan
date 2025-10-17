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
    git clone (https://github.com/Dwarvess/mnist_gan.git)
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd mnist_gan
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Test with the Pre-trained Model**
    The repository includes a pre-trained model (`generator_model5.pth`) and a folder of sample masked images (`/test_images`) for easy testing.

    You can test the model with any of the provided samples. For example, to test with `5_sayisi.png`, use the `--image` flag in the terminal:
    
    ```bash
    python train.py --image test_images/5_sayisi.png
    ```

    Simply replace `5_sayisi.png` with the filename of any other image located in the `/test_images` directory to see different results. You can also test the model with your own custom images by providing the correct path.

5.  **Train the Model (Optional)**
    To train the model from scratch with the optimal parameters, set `CONTROL_MODEL_FILE = 0` inside the `train.py` script and then run:
    ```bash
    python train.py
    ```

## 📈 Development Process & Results

The development process involved several iterations to find the optimal balance between visual quality (sharpness) and semantic correctness (not misinterpreting digits). The table below summarizes the key models developed.

### Analysis of Model Training Graphs

| Model Name | Generator Graph Analysis | Discriminator Graph Analysis | Technical Verdict |
| :--- | :--- | :--- | :--- |
| **Model 1** | **Healthy Generalization:** Training and validation losses track each other well. No overfitting. | **Hyper-Competitive/Unstable:** Very noisy and volatile. A healthy conflict exists, but the balance is delicate. | **Good but Risky** |
| **Model 2** | **Obvious Overfitting:** The validation loss plateaus very early. The model has stopped generalizing. | **Overly Stable:** The competition settles into an equilibrium too early, which can indicate that learning has slowed. | **Technically Flawed** |
| **Model 3 & 4**| **Early Overfitting:** The validation loss plateaus even earlier. The generalization ability is very weak. | **Stable Equilibrium:** Similar to Model 2, the competition is at a low level. | **Technically Flawed** |
| **🏆 Model 5 🏆**| **BEST GENERALIZATION:** The validation loss tracks the training loss for the longest duration. A slight tendency to overfit is present but under control. | **IDEAL COMPETITION:** Neither too unstable nor too stagnant. A healthy and stable equilibrium is present. | **TECHNICALLY SUPERIOR**|
| **Model 6** | **Unstable Overfitting:** The validation loss not only plateaus but is also very noisy and unstable. | **Unbalanced Competition:** A significant inconsistency exists between training and validation losses. The balance is broken. | **Technically Failed** |
