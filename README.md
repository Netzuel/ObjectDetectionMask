# Detection of persons using and not using masks in images, video and real time

The aim of this project is to use a **Keras** model in order to do detection of masks in images. This has been extended directly to videos since a video, as we know, can be understood just as a bunch of frames. In this repository I considered uploading my pre-trained model in its *.h5* (Keras) version and, as well, in *.tflite* (**Tensorflow Lite**) version. **Tensorflow Lite** is an extension of the **Tensorflow** framework which main purpose is using Tensorflow and Keras models in order to do inference in devices such as Android, iPhone, and IoT devices (you can see more information at its official webpage, clicking [here](https://www.tensorflow.org/lite?hl=es-419)).

We have used the **ImageAI** module for Python, which you can visite [here](https://imageai.readthedocs.io/en/latest/)). This module uses these packages and versions:

* NumPy == 1.16.1
* Keras == 2.2.4
* OpenCV
* Tensorflow == 1.13.1

Nevertheless, these versions are only required if you want to do transfer learning from YoloV3 in order to detect custom objects in your images, in case you want to detect objects that are not masks. This can be done easily visiting its webpage and following the different tutorials they have there, so I uploaded the pre-trained YoloV3 in case you want to use ImageAI (or do it by hand) to do your own custom object detection. This is actually what I have done in order to get the *model.h5* for Keras that is in this repository. The dataset used in order to **train** by transfer learning the YoloV3 model has been a bunch of 800+ photos of people wearing and not wearing masks, with a third label corresponding to weared incorrect mask. The dataset can be found in this Kaggle [link](https://www.kaggle.com/andrewmvd/face-mask-detection) (all rights saved for the uploader).

## Dependencies

This repository contains several Python scripts in order to make inference in images and videos. The dependencies are the following:

* NumPy == 1.18.1
* Tensorflow == 2.3.1
* Keras == 2.4.3
* OpenCV-Python == 4.4.0.46
* Matplotlib == 3.3.2 (Matplotlib is not necessary in order to execute the scripts, but *utilities.py* has two functions which use it in case you want to use them to draw the predictions).
* json == 2.0.9
* moviepy == 1.0.3
* tqdm == 4.51.0

In order to install them, if you do not have them already installed, first clone this repository:

```python
git clone https://github.com/Netzuel/ObjectDetectionMask
```

Then just move to the folder and write the following in the console:

```python
pip install -r requeriments.txt
```

or *pip3* depending on your configuration.

# Usage and examples

The general usage for detect masks in a image is the following:

```python
python ScriptDetectMask.py path_to_photo path_to_tflite_model path_to_json boolean_gpu threshold_detection
```

We explain them:

* **path_to_photo**: Here goes the path to the photo, as its name indicates. Remember to include the format, i.e. *.png*, *.jpg*, etc.
* **path_to_tflite_model**: Here goes the path to the file 'model.tflite' of this repository. This script in particular uses just TFLite and not Tensorflow in general so modifying it a little bit it should be possible to use it anyway with only having Tensorflow Lite installed.
* **path_to_json**: The path to the JSON file containing the anchors and labels for the model.
* **boolean_gpu**: This term should be literally *True* or *False* in case you want to use or not the GPU. By default the ID for the GPU device is indicated as 0, but this can be changed depending on your personal configuration. If that is the case, just modify the *0* in the line saying *os.environ["CUDA_VISIBLE_DEVICES"]="0"*. Remember that it is necessary to have CUDA installed and all the dependencies. If you dont have it, just put this term to *False* and just use CPU since for just an image detection it does not make the feel of a huge difference.
* **threshold_detection**: This is a number in the range (0,1], that is, it should be greater than zero and equal to one or less. This number indicates upper which we consider the detection and, obviously, detections with less probability would just be ignored.

Let's consider an example:

```python
python ScriptDetectMask.py people_normal.jpg models/model.tflite detection_config.json False 0.6
```

This will generate an *output.png* as output which is the same photo as input, but with the detected masks/no masks.

![Detection of masks](https://github.com/Netzuel/ObjectDetectionMask/blob/main/output1.png?raw=true)

We can consider another case where we have people wearing masks:

```python
python ScriptDetectMask.py people_with_masks.jpg models/model.tflite detection_config.json False 0.6
```

And we obtain:

![Detection of masks](https://github.com/Netzuel/ObjectDetectionMask/blob/main/output2.png?raw=true)

As we can see, we obtain the expected result. In addition, this can be also achieved in videos since a video is just a sequence of images. We can write:

```python
python ScriptDetectMaskVideo.py path_to_video path_to_model path_to_json boolean_gpu threshold_detection
```
The parameters here have the same meaning as before but this time the model needs to be the *.h5* since this script is not written in order to work with TFLite since Tensorflow is faster here. Although this code can also run without using GPU, it is recommended to use it since, otherwise, it could take tons of time to execute all code for a video of, for example, 5 minutes. As an example you can click on the image below which will redirect you to a YouTube video.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/-LlM7j8K59s/0.jpg)](http://www.youtube.com/watch?v=-LlM7j8K59s "Ti Amo - Money Heist")

By last, we can also take a look at the real time detection although even with GPU the FPS rate is not the best, but, however, it also works.

```python
python ScriptDetectMaskRealTime.py path_to_tflite_model path_to_json boolean_gpu threshold_detection
```

```python
python ScriptDetectMaskRealTimeNOLITE.py path_to_model path_to_json boolean_gpu threshold_detection
```

These two scripts are in reality the same just for the fact that the first one uses Tensorflow Lite and the second one uses raw Tensorflow; the last one is a little bit faster.
