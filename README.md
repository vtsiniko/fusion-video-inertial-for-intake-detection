
# Implementation of fusion networks for detection food intake cycle from video and inertial sensors. We use FICv dataset to evaluate the model


## Preprocess Data

1) Inertial <br />
-moving average filter <br />
-high pass FIR filter  

```
$ python inertial_preprocess.py
```


1) Video <br />
-face detection <br />
-crop backround <br />
-downsample 5 from 60 fps <br />
-low resolution to 128x128 <br />
-convert to greyscale

```
$ python video_preprocess.py
```

## Create window dataset for train 

```
$ python fusion_dataset.py
```

### What are the flags?

| Argument | Description |
| --- | --- |
| --time_window | Choose the default time for window data (5 second) |
| --model_fusion | Intermediate Fusion (IF) - will be added Late Fusion (LF)|



 
## Train Model and detect food intake cycle 
We use keras framework to train our model 


```
$ python fusion_train_for_LOSO.py
```


```
$ python fusion_prediction.py
```


### What are the flags?


| Argument | Description |
| --- | --- |
| --time_window | Choose the default time for window data  (seconds) |
| --model_fusion | Intermediate Fusion 'IF' - will be added Late Fusion 'LF'|
| --num_epochs | Choose the default number of epochs |
| --architectures | Choose the default network for the second channel (video data) between '3D_CNN' and '2D_CNN-LSTM'|



### Evaluate F1 score

```
$ python fusion_evaluation.py
```

### What are the flags?


| Argument | Description |
| --- | --- |
| --time_window | Choose the default time for window data  (seconds) |
| --model_fusion | Intermediate Fusion 'IF' - will be added Late Fusion 'LF'|
| --num_epochs | Choose the default number of epochs |
| --architectures | Choose the default network for the second channel (video data) between '3D_CNN' and '2D_CNN-LSTM'|
| --threshold | Replace with zeros elements of p that are lower than a probability threshold |
| --distance | Minimum distance between two consecutive peaks for local maxima search|


## Results on FICv dataset







Our models are trained on the [FICv dataset](https://mug.ee.auth.gr/intake-cycle-detection/), which is available to research groups.
The following models have been trained on the training set of 21 participants to detecet food intake cyckle.

F1 is based on actual detection of individual intake gestures on the test set for LOSO experiments.   

| Model | Features | Time_window  | Threshold  | Distance | F1 |
| --- | ---  | --- | --- | --- | --- |
| 1D_CNN-LSTM and 3D_CNN | inertial-video | 5sec | 0.97 | 3sec | 92.1% |
| 1D_CNN-LSTM and 2D_CNN-LSTM | inertial-video | 5sec | 0.98 | 3sec | 89.1% |


## Required 

-Tensorflow 2.4.3 <br />
-cv2 <br />
-face-recognition (https://pypi.org/project/face-recognition/)


