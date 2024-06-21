# Deep Learning Model Architectures

Welcome to the Deep Learning Model Architectures repository! This repository contains implementations of various deep learning model architectures. Each model is implemented in Python and organized into its own directory for ease of use and understanding.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Requirements](#requirements)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Deep learning has numerous applications, and understanding different model architectures is crucial for applying these techniques effectively. This repository provides implementations of several popular deep learning models, including AlexNet, InceptionNet, MobileNet, ResNet, and VGGNet. Each implementation is designed to be clear and educational.

## Installation

To get started, clone the repository and install the required dependencies:

git clone https://github.com/saketjha34/Deep-Learning-Model-Architectures.git
cd Deep-Learning-Model-Architectures
pip install -r requirements.txt


## Usage

Each model has its own directory containing the implementation and a sample script. To run a sample script, navigate to the model's directory and execute the script:

```bash
cd AlexNet
python AlexNet.py
```

You can modify the scripts and model parameters as needed for your specific use case.

## Model Architectures

This repository includes the following model architectures:

- [AlexNet](AlexNet/): Implementation of AlexNet, a pioneering deep convolutional neural network.
- [InceptionNet](InceptionNet/): Implementation of the Inception architecture, also known as GoogLeNet.
- [MobileNet](MobileNet/): Implementation of MobileNet, a lightweight deep learning model for mobile and embedded vision applications.
- [ResNet](ResNet/): Implementation of ResNet, known for its use of residual connections to ease the training of deep networks.
- [VGGNet](VGGNet/): Implementation of VGGNet, a deep convolutional network known for its simplicity and depth.
- [UNET](UNET/): A convolutional network architecture designed for biomedical image segmentation with a U-shaped structure enabling precise localization.
- [DCGANs](DCGANs/): A Deep Convolutional Generative Adversarial Network used for generating realistic images through adversarial training between a generator and a discriminator.

## Requirements

Make sure you have the following dependencies installed:
- Python 3.11.4
- NumPy
- PyTorch 
- Other dependencies as listed in `requirements.txt`

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Deployment

The trained models are saved as `.pth` files in the `pytorch saved models` directory. These files can be used for further deployment purposes. You can load the models in PyTorch using the following code:

```python
import torch
from ALexNet.AlexNet import AlexNet224
from InceptionNet.InceptionNetv1 import InceptionNetv1

# Load AlexNet model
alexnet = AlexNet224(in_channels = 3 , num_classes = 10)
alexnet.load_state_dict(torch.load('own trained model/load_your_own_trained_model.pth'))

# Load InceptionNetV1 model
inceptionnet = InceptionNetv1(in_channels = 3 , num_classes = 10)
inceptionnet.load_state_dict(torch.load(torch.load('own trained model/load_your_own_trained_model.pth'))
```

## References

If you use any part of this repository in your research, please cite the original papers:

- **AlexNet**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **VGGNet**: Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. [Paper](https://arxiv.org/abs/1409.1556)
- **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385. [Paper](https://arxiv.org/abs/1512.03385)
- **MobileNet**: Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861. [Paper](https://arxiv.org/abs/1704.04861)
- **InceptionNet**: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2014). Going deeper with convolutions. arXiv preprint arXiv:1409.4842. [Paper](https://arxiv.org/abs/1409.4842)
- **UNet**: "Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015). [Paper](https://arxiv.org/abs/1505.04597).
- **DCGANs**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz, and Soumith Chintala 2015. [Paper](https://arxiv.org/abs/1511.06434).


## Contributing

Contributions are welcome! If you have any improvements or new models to add, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your contributions follow the coding standards and include appropriate documentation and tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions, please open an issue or contact me directly at saketjha0324@gmail.com.

---

Happy coding!
