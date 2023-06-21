__MY_ROOT__=`pwd`
echo [$(date)]: "Started executing: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "START" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "add test.log.txt to gitignore" >> $__MY_ROOT__/test.log.txt
echo "test.log.txt" >> .gitignore
echo [$(date)]: "protobuff installation" >> $__MY_ROOT__/test.log.txt
pip install protobuf-compiler && protoc --version 
echo [$(date)]: "Create TensorFlow dir and cd to it" >> $__MY_ROOT__/test.log.txt
mkdir TensorFlow && cd TensorFlow
echo [$(date)]: "clone TensorFlow/models repo" >> $__MY_ROOT__/test.log.txt
git clone https://github.com/tensorflow/models.git
echo [$(date)]: "remove .git of models toi avoid conflict" >> $__MY_ROOT__/test.log.txt
rm -rf models/.git
echo [$(date)]: "add TensorFlow dir to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "TensorFlow/models" >> ../.gitignore
echo [$(date)]: "cd to research dir" >> $__MY_ROOT__/test.log.txt
cd ./models/research
echo [$(date)]: "convert protos to protobuff" >> $__MY_ROOT__/test.log.txt
protoc object_detection/protos/*.proto --python_out=.
echo [$(date)]: "copy setup to research dir" >> $__MY_ROOT__/test.log.txt
cp object_detection/packages/tf2/setup.py .
echo [$(date)]: "upgrade pip" >> $__MY_ROOT__/test.log.txt
pip install --upgrade pip --user
echo [$(date)]: "install object_detection api" >> $__MY_ROOT__/test.log.txt
python -m pip install .
echo [$(date)]: "Testing our installation" >> $__MY_ROOT__/test.log.txt
python object_detection/builders/model_builder_tf2_test.py
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__
# -----------------------------------
echo [$(date)]: "START workspace" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "create workspace training demo" >> $__MY_ROOT__/test.log.txt
mkdir -p workspace/training_demo
echo [$(date)]: "cd to workspace/training_demo" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "create workspace" >> $__MY_ROOT__/test.log.txt
mkdir -p annotations exported-models models/my_ssd_resnet50_v1_fpn pre-trained-models images
echo [$(date)]: "create label map in annotations dir with classes" >> $__MY_ROOT__/test.log.txt
echo "item {
    id: 1
    name: 'helmet'
}
item {
    id: 2
    name: 'head'
}
item {
    id: 3
    name: 'person'
}" > annotations/label_map.pbtxt
echo [$(date)]: "curl tfrecord converter in the root of workspace/training_demo" >> $__MY_ROOT__/test.log.txt
curl https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py > generate_tfrecord.py
echo [$(date)]: "use labelImg to do the annotations" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__
# --------------------------------

echo [$(date)]: "START" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "unzip dataset" >> $__MY_ROOT__/test.log.txt
unzip 'Hard Hat Workers.v2-raw.voc.zip'
echo [$(date)]: "remove unwanted txt files" >> $__MY_ROOT__/test.log.txt
rm README.dataset.txt README.roboflow.txt
echo [$(date)]: "mv train dir to images" >> $__MY_ROOT__/test.log.txt
mv train workspace/training_demo/images
echo [$(date)]: "mv test dir to images" >> $__MY_ROOT__/test.log.txt
mv "test" workspace/training_demo/images
echo [$(date)]: "add train to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "workspace/training_demo/images/train/*" >> .gitignore
echo [$(date)]: "add test to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "workspace/training_demo/images/test/*" >> .gitignore
echo [$(date)]: "cd to training_demo dir" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "generate train tfrecord" >> $__MY_ROOT__/test.log.txt
python generate_tfrecord.py -x images/train -l annotations/label_map.pbtxt -o annotations/train.record
echo [$(date)]: "generate test tfrecord" >> $__MY_ROOT__/test.log.txt
python generate_tfrecord.py -x images/test -l annotations/label_map.pbtxt -o annotations/test.record
echo [$(date)]: "add misc files to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "*.record" >> ../../.gitignore
echo "*.checkpoint" >> ../../.gitignore
echo "*.pb" >> ../../.gitignore
echo "variable*" >> ../../.gitignore
echo [$(date)]: "cd to pre-trained-models" >> $__MY_ROOT__/test.log.txt
cd pre-trained-models/
echo [$(date)]: "curl ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 model" >> $__MY_ROOT__/test.log.txt
curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz > ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 
echo [$(date)]: "add .gz file to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz" >> ../../../.gitignore
echo [$(date)]: "untar gz file" >> $__MY_ROOT__/test.log.txt
tar -xzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__

echo [$(date)]: "Change to training_demo" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "Create models/my_ssd_resnet50_v1_fpn" >> $__MY_ROOT__/test.log.txt
mkdir -p models/my_ssd_resnet50_v1_fpn
echo [$(date)]: "echo config file" >> $__MY_ROOT__/test.log.txt
echo 'model {
  ssd {
    num_classes: 3
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "ssd_resnet50_v1_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00039999998989515007
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }
      }
      override_base_feature_extractor_hyperparams: true
      fpn {
        min_level: 3
        max_level: 7
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00039999998989515007
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.009999999776482582
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }
        }
        depth: 256
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          gamma: 2.0
          alpha: 0.25
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }
}
train_config {
  batch_size: 4
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03999999910593033
          total_steps: 25000
          warmup_learning_rate: 0.013333000242710114
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 25000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  use_bfloat16: false
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "annotations/train.record"
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "annotations/test.record"
  }
}' > models/my_ssd_resnet50_v1_fpn/pipeline.config
echo [$(date)]: "cp training file to training_demo dir" >> $__MY_ROOT__/test.log.txt
cp ../../TensorFlow/models/research/object_detection/model_main_tf2.py .
echo [$(date)]: "Start Training.." >> $__MY_ROOT__/test.log.txt
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
echo [$(date)]: "copy model exporter to training_demo dir and run this" >> $__MY_ROOT__/test.log.txt
cp ../../TensorFlow/models/research/object_detection/exporter_main_v2.py .
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model
echo [$(date)]: "inferencing the model and run it for testing." >> $__MY_ROOT__/test.log.txt
echo '#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel("ERROR")           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMAGE_PATHS = ["./images/test/"] ### << GIVE PATH OF TEST IMAGES


# %%
# Download the model
# ~~~~~~~~~~~~~~~~~~
# The code snippet shown below is used to download the pre-trained object detection model we shall
# use to perform inference. The particular detection algorithm we will use is the
# `CenterNet HourGlass104 1024x1024`. More models can be found in the `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
# To use a different model you will need the URL name of the specific model. This can be done as
# follows:
#
# 1. Right click on the `Model name` of the model you would like to use;
# 2. Click on `Copy link address` to copy the download link of the model;
# 3. Paste the link in a text editor of your choice. You should observe a link similar to ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
# 4. Copy the ``XXXXXXXXX`` part of the link and use it to replace the value of the ``MODEL_NAME`` variable in the code shown below;
# 5. Copy the ``YYYYYYYY`` part of the link and use it to replace the value of the ``MODEL_DATE`` variable in the code shown below.
#
# For example, the download link for the model used below is: ``download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz``

# Download and extract model

PATH_TO_MODEL_DIR = "./exported-models/my_model" ## << Directory of saved model

# %%
# Download the labels
# ~~~~~~~~~~~~~~~~~~~
# The coode snippet shown below is used to download the labels file (.pbtxt) which contains a list
# of strings used to add the correct label to each detection (e.g. person). Since the pre-trained
# model we will use has been trained on the COCO dataset, we will need to download the labels file
# corresponding to this dataset, named ``mscoco_label_map.pbtxt``. A full list of the labels files
# included in the TensorFlow Models Garden can be found `here <https://github.com/tensorflow/models/tree/master/research/object_detection/data>`__.

# Download labels file
PATH_TO_LABELS = "annotations/label_map.pbtxt"

# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print("Loading model...", end="")
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))

# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.functions trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections["detection_boxes"]` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print("Running inference for {}... ".format(image_path), end=" ")

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We are only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["etection_classes"].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections["detection_boxes"],
          detections["detection_classes"],
          detections["detection_scores"],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print("Done")
plt.show() ' > ./inferencing.py
python inferencing.py
echo [$(date)]: "Execution complete for: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: ">>>>>>>>> END <<<<<<<<<" >> $__MY_ROOT__/test.log.txt
