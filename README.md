# Breast Cancer Classification With PyTorch
Every year in the US, around 180,000 people find out they have a serious type of breast cancer. The American Cancer Society says that 80% of these cases are invasive ductal carcinoma (IDC). IDC starts in the tubes that carry milk to the nipple and can spread to the nearby breast tissue. It usually happens more in older women, but sometimes men can get it too.
In this simple project, we are going to show how to use PyTorch and deep learning to sort and understand images of breast tissue to spot cancer. **It's important to know that figuring out the different kinds of breast cancer is a vital job that can take trained doctors hours to do**. Here, we plan to make this process faster by using different ways to automatically identify breast cancer types through these images with the help of PyTorch and deep learning.

## The Dataset

Breast histopathology images can be downloaded from [Here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
These images were put together by [Janowczyk and Madabhushi and Roa et al.](https://pubmed.ncbi.nlm.nih.gov/27563488/) There are 227,524 small sections, each 50 x 50 pixels, taken from 162 large images of breast cancer samples scanned at high magnification (40x). The collection includes both types of images: those without cancer (called benign) and those with cancer (called malignant). However, there are more than twice as many images without cancer as there are with cancer, showing that the data set has a lot more of one type than the other, which is a common issue called **imbalanced data**:

-   Images without cancer (benign): 198,738.
-   Images with cancer (malignant): 78,786.

## 00. A Classification Example Using a Simple CNN (with CIFAR-10 dataset)

In this Jupyter Notebook we use a simple CNN architecture to demonstrate a simple solution to a classification problem with 10 classes as output. For this demonstration, a basic dataset, **CIFAR-10**, was used.

## 01. Breast Cancer Data Processing

In this Jupyter Notebook we explain how to read and split the image data into training, validation, and testing sets. Like mentioned before, we have about 227,524 images in total. Out of these, we set aside 38,000 images for validation and another 38,000 for testing. We used the rest, which is 201,524 images, to train the model. Here's how it was divided:

-   Training: 201,524 images.
-   Validation: 38,000 images.
-   Testing: 38,000 images.

## 02. Data Augmentation

We have to manage an unbalanced dataset so, in this Notebook we develop a series of augmentations to an image aiming to increase the diversity of your training set:

- Rotation (90, 180, 270 degrees).
- Color enhancement (reducing color intensity).
- Contrast enhancement (increasing contrast).
- Flipping (left-right and top-bottom).

## 03. Logistic Regression (with the Imbalanced Dataset)

Logistic regression can serve as a straightforward and interpretable baseline model. Before trying more complex methods, it's helpful to establish a baseline performance with a simple model like logistic regression. This allows you to understand the basic patterns in the data and set a performance benchmark for more complex algorithms.

- `nn.BCEWithLogitsLoss()` is a good choice for binary (and multi-label) classification problems, especially when working with logits, dealing with **class imbalance**, or needing a stable and efficient way to combine sigmoid activation with binary cross-entropy loss.
- **Precision, recall, and the F1 score** offer a more nuanced understanding of a model's performance on imbalanced datasets.

## 04. A Simple CNN Architecture (with the Imbalanced Dataset)

In this Jupyter Notebook we create a simple architecture of a CNN that we train with the original imbalanced dataset. To address the issue of training a model on an imbalanced dataset in PyTorch we use using a weighted sampling technique. We calculate the inverse of the class counts to determine the weights for each class. The idea is to give more weight to the class with fewer instances and less weight to the class with more instances. Every instance in the training dataset is assigned a weight based on its class. 
The `WeightedRandomSampler` will be used to sample instances from the dataset during training. By providing the `samples_weights`, the sampler is informed how frequently an instance should be sampled. The second argument, `len(samples_weights)`, specifies the number of samples to draw, which in this case is the total number of instances in the training dataset. This ensures that the model sees instances from the minority class more often than it would in a purely random sampling approach, helping to balance the training process.

> class_counts = torch.tensor([144435, 57089])  
> class_weights = 1. / class_counts  
> samples_weights = torch.tensor([class_weights[t] for t in train_dataset.targets])  
> sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

## 05. Transfer Learning (Using the Balanced Dataset)

Transfer learning involves taking a model that has been trained on a large dataset and fine-tuning it for a specific task. Using TL with a **pretrained ResNet34** model offers a practical and efficient approach to achieving high-performance results on image classification tasks, especially when dealing with limited data or seeking to reduce development and training time.
In this task, we modify the final fully connected layer for binary classification. 
Loss function and optimizer used (for simplicity):

> criterion = nn.CrossEntropyLoss()  
> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

### Notes

- Simple values for hyperparameters have been used for demonstration. 
- Paths for the dataset must be adjusted accordingly.
- Packages:

> PyTorch:  2.1.0
> torchvision:  0.16.0
> matplotlib:  3.8.0
> PIL:  10.2.0
> numpy:  1.26.0
> sklearn:  1.4.1

### Further Steps

- Hyperparameter optimization using [optuna](https://neptune.ai/) (recommend).
- Track, organize, and analyze machine learning experiments using [Neptune.ai](https://neptune.ai/) (recommend).
