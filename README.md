DeepLogo
---
A brand logo detection system using tensorflow object detection API.

### Examples

Belows are detection examples.

![example1](detect_results/detect_result_029.png)
![example2](detect_results/detect_result_049.png)
![example3](detect_results/detect_result_055.png)
![example4](detect_results/detect_result_056.png)
![example5](detect_results/detect_result_082.png)
![example6](detect_results/detect_result_351.png)


### Usage

1. Setup the tensorflow object detection API. First of all, 
   clone the tensorflow/models repository. 
   ```
   $ git clone https://github.com/tensorflow/models.git
   $ cd models/research/object_detection
   $ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
   $ tar zxvf ssd_inception_v2_coco_2018_01_28.tar.gz
   ```
   For detailed steps to setup, please follow the [installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
2. Clone the DeepLogo repository.
   ```
   $ git clone https://github.com/satojkovic/DeepLogo.git
   ```
3. Download dataset from [flickr_27_logos_dataset](http://image.ntua.gr/iva/datasets/flickr_logos/) and extract.
   ```
   $ cd DeepLogo
   $ wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
   $ tar zxvf flickr_logos_27_dataset.tar.gz
   $ cd flickr_logos_27_dataset
   $ tar zxvf flickr_logos_27_dataset_images.tar.gz
   $ cd ../
   ```
4. Preprocess original annotation file and generate <u>flickr_logos_27_dataset_training_set_annotation_cropped.txt</u> and <u>flickr_logos_27_dataset_test_set_annotation_cropped.txt</u>. These two files are used to generate tfrecord files.
   ```
   $ cd DeepLogo
   $ python preproc_annot.py
   ```
5. Generate tfrecord files.
   ```
   $ python gen_tfrecord.py --csv_input flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation_cropped.txt --img_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path train.tfrecord
   $ python gen_tfrecord.py --csv_input flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt --img_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path test.tfrecord
   ```
6. Training logo detector using pre-trained SSD.
   ```
   $ python <OBJECT_DETECTION_API_DIR>/legacy/train.py --logtostderr --pipeline_config_path=ssd_inception_v2.config --train_dir=training
   ```
   <OBJECT_DETECTION_API_DIR> is the absolute path of models/research/object_detection at step1.

7. Testing logo detector.  
   ```
   $ python logo_detection.py --model_name logos_inference_graph/ --label_map flickr_logos_27_label_map.pbtxt --test_annot_text flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt --test_image_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_dir detect_results
   ```

### License

MIT
