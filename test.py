from time import time
import click
import torch
from PIL import Image
import numpy as np
import model
import eval
from torchvision import transforms
from torch.functional import F

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])


flow = model.UNet(6, 4).to(device)
interp = model.UNet(20, 5).to(device)
back_warp = None

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])



def make_frames(I0, I1, dest, factor, output_format = 'png'):


    I0 = Image.open(I0)
    I1 = Image.open(I1)
    assert I0.size ==  I1.size

    I0 = I0.convert('RGB')
    I1 = I1.convert('RGB')

    w0, h0 = I0.size
    w, h = (w0 // 32) * 32, (h0 // 32) * 32

    #setup the back warping function in model
    eval.setup_back_warp(w, h)

    #image to tensor
    I0 = I0.resize((w, h), Image.ANTIALIAS)
    I1 = I1.resize((w, h), Image.ANTIALIAS)
    I0 = trans_forward(I0)
    I1 = trans_forward(I1)

    batch = []
    batch.append(I0)
    batch.append(I1)

    intermediate_frames = eval.interpolate_batch(batch, factor)
    intermediate_frames = list(zip(*intermediate_frames))

    intermediate_frames = intermediate_frames[0]

    i = 0
    for img in intermediate_frames:
        x = trans_backward(img)
        x = x.resize((w0, h0), Image.BILINEAR)
        x = x.convert('RGB')
        x.save(dest+str(i)+'.'+output_format)
        i = i+1

@click.command('Evaluate Model by converting a low-FPS video to high-fps')
@click.argument('input1')
@click.argument('input2')
@click.option('--checkpoint', help='Path to model checkpoint')
@click.option('--output', help='Path to output file to save')
@click.option('--scale', default=4, help='Scale Factor of FPS')
def main(input1, input2, checkpoint, output, scale):
    load_models(checkpoint)
    make_frames(input1, input2, output, int(scale))



if __name__ == '__main__':
    main()
