DeepLogo
---
A brand logo detection system using region-based convolutional neural networks.

### Overview

Implentation of Region-based Convolutional Neural Networks in Tensorflow, to detect and classify brand logos.

Note: Since this is my first version model, there are some remaining issues.

Example detection results are below.

![example1](query_set_results/3666600356_Cocacola.png)
![example2](query_set_results/4273898682_DHL.png)
![example3](query_set_results/3907703753_Fedex.png)
![example4](query_set_results/6651198_McDonalds.png)
![example5](query_set_results/401253895_BMW.png)
![example6](query_set_results/2180367311_Google.png)

Here are some failure cases.

![example7](query_set_results/4288066623_Unicef.png)


### Usage

1. `python gen_bg_class.py`: Generate train\_annot\_with\_bg\_class.txt file. 
2. `python crop_and_aug.py`: Crop brand logo images from the [flickr27\_logos\_dataset](http://image.ntua.gr/iva/datasets/flickr_logos/) and apply data augmentation method. Finally the dataset consists of 140137 images.
3. `python gen_train_valid_test.py`: Generate(Split) train/valid/test set from the dataset.
4. `python train_deep_logo_cnn.py`: Train the convolutional neural networks and save the trained model to disk.
5. `python test_deep_logo_cnn.py`: Test the trained model (for Classification).  
`python detect_logo.py`: Test the trained model (for Detection)

### Network

The network is based on [this blog post](http://matthewearl.github.io/2016/05/06/cnn-anpr/). Same network is applied to this brand logo recognition task because a brand logo is similar to a number plate which consists of a number of digits and letters.

### License

MIT
