from time import time
import click
import torch
from PIL import Image
import numpy as np
import model
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


def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])

def interpolate_batch(frames, factor):
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)

    flow_out = flow(ix)
    f01 = flow_out[:, :2, :, :]
    f10 = flow_out[:, 2:, :, :]

    frame_buffer = []
    for i in range(1, factor):
        t = i / factor
        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10
        ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = F.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)
        gi1ft1f = back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)

        frame_buffer.append(ft_p)

    return frame_buffer

def make_frames(I0, I1, dest, factor, output_format = 'png'):


    I0 = Image.open(I0)
    I1 = Image.open(I1)
    assert I0.size ==  I1.size

    I0 = I0.convert('RGB')
    I1 = I1.convert('RGB')

    w0, h0 = I0.size
    w, h = (w0 // 32) * 32, (h0 // 32) * 32

    #setup the back warping function in model
    setup_back_warp(w, h)

    #image to tensor
    I0 = I0.resize((w, h), Image.ANTIALIAS)
    I1 = I1.resize((w, h), Image.ANTIALIAS)
    I0 = trans_forward(I0)
    I1 = trans_forward(I1)

    batch = []
    batch.append(I0)
    batch.append(I1)

    intermediate_frames = interpolate_batch(batch, factor)
    intermediate_frames = list(zip(*intermediate_frames))

    #only one pair of image, len(intermadiated_frams) == number of the frame pairs
    intermediate_frames = intermediate_frames[0]

    i = 0
    for img in intermediate_frames:
        x = img.cpu()
        x = trans_backward(x)
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
