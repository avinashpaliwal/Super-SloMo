import argparse
import os
import os.path
from shutil import rmtree, move
import random

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
    for video in videos:
        os.mkdir(os.path.join(outDir, os.path.splitext(video)[0]))
        retn = os.system('{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), os.path.join(inDir, video), args.img_width, args.img_height, os.path.join(outDir, os.path.splitext(video)[0])))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))

def create_clips(root, destination):
    folderCounter = -1

    files = os.listdir(root)
    for file in files:
        images = os.listdir(os.path.join(root, file))

        for imageCounter, image in enumerate(images):
            if (imageCounter % 12 == 0):
                if (imageCounter + 11 >= len(images)):
                    break
                folderCounter += 1
                os.mkdir("{}/{}".format(destination, folderCounter))
            move("{}/{}/{}".format(root, file, image), "{}/{}/{}".format(destination, folderCounter, image))
        rmtree(os.path.join(root, file))

def main():
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

        testClips = os.listdir(testPath)
        indices = random.sample(range(len(testClips)), 100)
        for index in indices:
            move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

    else: # custom dataset
        #TBD
        pass

    rmtree(extractPath)

main()