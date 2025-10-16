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



TRAIN_MASK_MODE = "mixed"
This parameter determines the type of mask applied to images during training. Available options include "horizontal", "vertical", "random_square", and "mixed". 
In my opinion, the "mixed" mode is the most effective as it trains the model to be robust against various types of occlusions.

TRAIN_SQUARE_SIZE = 
If you select "random_square" as the mask mode, this value sets the side length of the square mask.

TRAIN_LINE_THICKNESS = 
If you choose "horizontal" or "vertical" as the mask mode, this value determines the thickness of the line mask.

BATCH_SIZE = 
This defines the number of samples processed in a single iteration of training. A larger batch size can lead to a more stable and accurate estimate of the gradient,
but it also increases memory consumption and can sometimes lengthen the total training time.

NUM_EPOCHS = 
An epoch represents one complete pass of the entire training dataset through the model. This parameter specifies the total number of epochs for training. Too few epochs
may result in an undertrained model (underfitting), while too many can cause the model to memorize the training data and perform poorly on new data (overfitting).

LR = 
The Learning Rate (LR) controls the step size the optimizer takes during gradient descent. There is often a relationship between batch size and learning rate; if you
decrease the batch size, you may need to decrease the learning rate as well.

A high LR can cause the model to converge too quickly to a suboptimal solution or even diverge.

A low LR can make the training process excessively slow.
Finding an optimal LR is crucial for effective training.

LAMBDA_RECON = 
This is a weight coefficient that determines the importance of the Reconstruction Loss (in this case, L1 Loss). It's a hyperparameter used to balance the two main 
objectives of the generator: creating realistic images (driven by the GAN loss) and ensuring the output is faithful to the original, unmasked image (driven by the
reconstruction loss). A high value, like 1000, strongly encourages the model to prioritize pixel-perfect reconstruction.

CONTROL_MODEL_FILE = 
This flag controls whether to use a pre-trained model or to train a new one from scratch.

If set to 1, the script will load the existing trained model from the MODEL_PATH.

If set to 0, the script will delete any existing model file and restart the training process.

test_custom_image(r"D:\cods\Data.png")
This is the function call to test the trained model with your own custom image. You should change the file path to point to the location of your test image.
