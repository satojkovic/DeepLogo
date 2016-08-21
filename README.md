DeepLogo
---
A brand logo recognition system using deep convolutional neural networks.

### Usage

1. `python crop_and_aug.py`: Crop brand logo images from the [flickr27_logos_dataset](http://image.ntua.gr/iva/datasets/flickr_logos/) and apply data augmentation method. Finally the dataset consists of 217488 images.
2. `python gen_train_valid_test.py`: Generate(Split) train/valid/test set from the dataset.
3. `python train_deep_logo_cnn.py`: Train the convolutional neural networks and save the trained model to disk.
4. `python test_deep_logo_cnn.py`: Test the trained model.

### Network

The network is based on [this blog post](http://matthewearl.github.io/2016/05/06/cnn-anpr/). Same network is applied to this brand logo recognition task because a brand logo is similar to a number plate which consists of a number of digits and letters.

### License

MIT
