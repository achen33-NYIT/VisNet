## A. Introduction

Convolutional Neural Networks (CNNs) have revolutionized the field of image recognition, setting new benchmarks in accuracy and efficiency. A CNN's unique architecture enables it to automatically and adaptively learn spatial hierarchies of features from images. This learning process is done through a series of convolutional layers that mimic the human visual cortex, making CNNs exceptionally good at recognizing visual patterns directly from pixel images with minimal preprocessing

Convolutional Neural Networks (CNNs) have set new benchmarks in image recognition by learning spatial hierarchies of features from images. Among notable architectures, VGGNet16, GoogLeNet, and ResNet stand out. VGGNet16's deep architecture offers robust feature representation but is computationally intensive. GoogLeNet, with its inception blocks, reduces computational costs while maintaining high accuracy. ResNet addresses vanishing gradients with residual learning, enabling deeper networks.

After evaluating these models, GoogLeNet was selected for our project due to its optimal balance of efficiency and accuracy, achieving 78.70% accuracy on our tests, making it ideal for real-world image recognition tasks where resource balance is crucial.


## B. Setup Instructions

### Using Google Colab

1. **Open Google Colab:**
   - Visit [Google Colab](https://colab.research.google.com/).
   - Click on `File -> New notebook`.

2. **Configure Runtime:**
   - Go to `Runtime -> Change runtime type`.
   - Select `Python 3` as the runtime type.
   - Select `GPU` as the hardware accelerator.
   - Click `Save`.

3. **Install PyTorch:**
   ```python
   !pip install torch torchvision
   ```

4. **Check PyTorch Version:**
   ```python
   import torch
   print(torch.__version__)
   ```

5. **Mount Google Drive (if needed):**
   ```python
   from google.colab import drive
   drive.mount('/gdrive')
   ```

### Using Jupyter Notebook

1. **Set Up Virtual Environment:**
   - For Windows:
     ```bash
     py -3 -m venv env
     .\env\scripts\activate
     ```
   - For Mac/Linux:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```

2. **Install Required Packages:**
   ```bash
   pip install torch torchvision
   ```

3. **Open Jupyter Notebook:**
   - Run the command:
     ```bash
     jupyter notebook
     ```

4. **Create a New Notebook:**
   - Navigate to the Jupyter Notebook interface in your browser.
   - Click `New -> Python 3` to create a new notebook.

5. **Check PyTorch Version:**
   ```python
   import torch
   print(torch.__version__)
   ```

6. **Run Your Code:**
   - Write and execute your CNN training and evaluation code in the notebook cells.




## C. Training Results

![Alt text](<CNN_Model_Screenshots/VGGNet16.png> "CNN Model 1")

**CNN Model 1: VGGNet16**

Training Accuracy: Started at 53.50%, improved to 59.20% over 5 epochs. Test Accuracy: Started at 52.66%, improved to 56.56%. Training Time per Epoch: ~45 seconds.


![Alt text](<CNN_Model_Screenshots/GoogLeNet.png> "CNN Model 2")

**CNN Model 2: GoogLeNet**

Training Accuracy: Started at 78.11%, improved to 78.70% over 5 epochs. Test Accuracy: Started at 75.15%, improved to 77.28%. Training Time per Epoch: ~98 seconds.

![Alt text](<CNN_Model_Screenshots/ResNet.png> "CNN Model 3")

**CNN Model 3: ResNet**

Training Accuracy: Started at 51.37%, improved to 53.18% over 5 epochs. Test Accuracy: Started at 45.02%, improved to 51.38%. Training Time per Epoch: ~20 seconds.

## D. GoogleNet Selected for CIFAR-10 dataset

The performances of these CNNs have been meticulously evaluated on various datasets. For VGGNet16, the image recognition capabilities, although robust, were overshadowed by the computational demands of the model. ResNet's introduction of residual blocks significantly improved training efficacy and accuracy, especially in deeper networks. However, it was GoogLeNet's innovative inception blocks that provided an optimal balance of computational efficiency and high accuracy, leading to its selection for our project.

With the training results in hand, GoogLeNet has been chosen as the preferred model, achieving a leading accuracy of 78.70%. This underscores GoogLeNet's capability to efficiently handle complex image recognition tasks, making it an excellent choice for real-world applications where balance between accuracy and computational resources is paramount.