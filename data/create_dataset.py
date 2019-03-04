import argparse
import os
import os.path
from shutil import rmtree, move
import random

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, required=True, help='path to ffmpeg.exe')
parser.add_argument("--dataset", type=str, default="custom", help='specify if using "adobe240fps" or custom video dataset')
parser.add_argument("--videos_folder", type=str, required=True, help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--img_width", type=int, default=640, help="output image width")
parser.add_argument("--img_height", type=int, default=360, help="output image height")
parser.add_argument("--train_test_split", type=tuple, default=(90, 10), help="train test split for custom dataset")
args = parser.parse_args()


def extract_frames(videos, inDir, outDir):
    """
    Converts all the videos passed in `videos` list to images.

    Parameters
    ----------
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        None
    """


    for video in videos:
        os.mkdir(os.path.join(outDir, os.path.splitext(video)[0]))
        retn = os.system('{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), os.path.join(inDir, video), args.img_width, args.img_height, os.path.join(outDir, os.path.splitext(video)[0])))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))


def create_clips(root, destination):
    """
    Distributes the images extracted by `extract_frames()` in
    clips containing 12 frames each.

    Parameters
    ----------
        root : string
            path containing extracted image folders.
        destination : string
            path to output clips.

    Returns
    -------
        None
    """


    folderCounter = -1

    files = os.listdir(root)

    # Iterate over each folder containing extracted video frames.
    for file in files:
        images = sorted(os.listdir(os.path.join(root, file)))

        for imageCounter, image in enumerate(images):
            # Bunch images in groups of 12 frames
            if (imageCounter % 12 == 0):
                if (imageCounter + 11 >= len(images)):
                    break
                folderCounter += 1
                os.mkdir("{}/{}".format(destination, folderCounter))
            move("{}/{}/{}".format(root, file, image), "{}/{}/{}".format(destination, folderCounter, image))
        rmtree(os.path.join(root, file))

def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    extractPath      = os.path.join(args.dataset_folder, "extracted")
    trainPath        = os.path.join(args.dataset_folder, "train")
    testPath         = os.path.join(args.dataset_folder, "test")
    validationPath   = os.path.join(args.dataset_folder, "validation")
    os.mkdir(extractPath)
    os.mkdir(trainPath)
    os.mkdir(testPath)
    os.mkdir(validationPath)

    if(args.dataset == "adobe240fps"):
        f = open("adobe240fps/test_list.txt", "r")
        videos = f.read().split('\n')
        extract_frames(videos, args.videos_folder, extractPath)
        create_clips(extractPath, testPath)

        f = open("adobe240fps/train_list.txt", "r")
        videos = f.read().split('\n')
        extract_frames(videos, args.videos_folder, extractPath)
        create_clips(extractPath, trainPath)

        # Select 100 clips at random from test set for validation set.
        testClips = os.listdir(testPath)
        indices = random.sample(range(len(testClips)), 100)
        for index in indices:
            move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

    else: # custom dataset
        
        # Extract video names
        videos = os.listdir(args.videos_folder)

        # Create random train-test split.
        testIndices  = random.sample(range(len(videos)), int((args.train_test_split[1] * len(videos)) / 100))
        trainIndices = [x for x in range((len(videos))) if x not in testIndices]

        # Create list of video names
        testVideoNames  = [videos[index] for index in testIndices]
        trainVideoNames = [videos[index] for index in trainIndices]

        # Create train-test dataset
        extract_frames(testVideoNames, args.videos_folder, extractPath)
        create_clips(extractPath, testPath)
        extract_frames(trainVideoNames, args.videos_folder, extractPath)
        create_clips(extractPath, trainPath)

        # Select clips at random from test set for validation set.
        testClips = os.listdir(testPath)
        indices = random.sample(range(len(testClips)), min(100, int(len(testClips) / 5)))
        for index in indices:
            move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

    rmtree(extractPath)

main()
