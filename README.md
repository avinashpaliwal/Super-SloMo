# Super-SloMo [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang et al. [[Project]](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) [[Paper]](https://arxiv.org/abs/1712.00080)

## Results
<img src='./misc/original.gif'>
<img src='./misc/slomo.gif'>

## Prerequisites
This codebase was developed and tested with pytorch 0.4.1 and CUDA 9.2.

## Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner.
The create_dataset.py script uses [ffmpeg](https://www.ffmpeg.org/) to extract frames from videos.
For adobe240fps, [download the dataset](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip), unzip it and then run the following command
```bash
python data\create_dataset.py --ffmpeg_dir path\to\ffmpeg --videos_folder path\to\adobe240fps\videoFolder --dataset_folder path\to\dataset --dataset adobe240fps
```

## To-Do's:
|Add evaluation script for UCF dataset | TBD|  
|Add pretrained model | In Progress|  
|Add getting started guide | TBD|  
|Add video converter script | In progress|  