# Super-SloMo [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J. [[Project]](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) [[Paper]](https://arxiv.org/abs/1712.00080)

## Results
Results on UCF101 dataset using the [evaluation script](https://people.cs.umass.edu/~hzjiang/projects/superslomo/UCF101_results.zip) provided by paper's author. The `get_results_bug_fixed.sh` script was used. It uses motions masks when calculating PSNR, SSIM and IE.  

| Method | PSNR | SSIM | IE |
|------|:-----:|:-----:|:-----:|
| DVF | 29.37 | 0.861 | 16.37 |  
| [SepConv](https://github.com/sniklaus/pytorch-sepconv) - L_1 | 30.18 | 0.875 | 15.54 |  
| [SepConv](https://github.com/sniklaus/pytorch-sepconv) - L_F | 30.03 | 0.869 | 15.78 |  
| SuperSloMo_Adobe240fps | 29.80 | 0.870 | 15.68 |  
| **pretrained mine** | **29.77** | **0.874** | **15.58** |  
| SuperSloMo | 30.22 | 0.880 | 15.18 |  


<img src='./misc/original.gif'>
<img src='./misc/slomo.gif'>

## Prerequisites
This codebase was developed and tested with pytorch 0.4.1 and CUDA 9.2.

## Training
### Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner.  
The create_dataset.py script uses [ffmpeg](https://www.ffmpeg.org/) to extract frames from videos.  
For adobe240fps, [download the dataset](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip), unzip it and then run the following command
```bash
python data\create_dataset.py --ffmpeg_dir path\to\ffmpeg --videos_folder path\to\adobe240fps\videoFolder --dataset_folder path\to\dataset --dataset adobe240fps
```

### Training
In the [train.ipynb](train.ipynb), set the parameters (dataset path, checkpoint directory, etc.) and run all the cells.  

### Tensorboard
To get visualization of your training, you can run tensorboard from the project directory using the command:
```bash
tensorboard --logdir logs --port 6007
```

## Evaluation
### Pretrained model
You can download the pretrained model trained on adobe240fps dataset [here](https://drive.google.com/open?id=1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF).

More info TBA

## To-Do's:
| Task | Status |
|------|--------|
|Add evaluation script for UCF dataset | TBD|  
|Add getting started guide | TBD|  
|Add video converter script | In progress|  
