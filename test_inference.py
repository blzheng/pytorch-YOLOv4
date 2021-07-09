import torch
import argparse
import os
import random
import shutil
import time
import warnings
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
from yolo_loss import Yolo_loss
from torch.utils import mkldnn as mkldnn_utils
from models import Yolov4
from dataset import Yolo_dataset

parser = argparse.ArgumentParser(description='PyTorch Yolov4 Training')
parser.add_argument('-N', '--n_classes', default=80, type=int, metavar='n_classes',
                    help='num classes')
parser.add_argument('-w', '--weightfile', type=str, default='./yolov4.pth',
                    help='weight file')
parser.add_argument('-i', '--imgfile', type=str, default='./data/dog.jpg',
                    help='image file')
parser.add_argument('--height', default=320, type=int, help='height')
parser.add_argument('--width', default=320, type=int, help='width')
parser.add_argument('-n', '--namesfile', type=str, help='names file')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--int8', action='store_true', default=False,
                    help='enable ipex int8 path')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable ipex bf16 path')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
parser.add_argument('--calibration', action='store_true', default=False,
                    help='doing calibration step for int8 path')
parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH',
                    help = 'path to int8 configures, default file name is configure.json')
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--warmup', default=30, type=int, metavar='N',
                    help='number of warmup iterati ons to run')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

def fp32_imperative_stock_pytorch(model, images):
    model = model.to(memory_format=torch.channels_last)
    images = images.to(memory_format=torch.channels_last)
    output=model(images)
    print(output)

def fp32_jit_stock_pytorch(model, images, args):
    model = torch.jit.trace(model, torch.randn(args.batch_size, 3, args.height, args.width))
    model = torch.jit.freeze(model)
    images = images.to(memory_format=torch.channels_last)
    output=model(images)
    print(output)

def bf16_imperative_stock_pytorch(model, images):
    model = model.to(torch.bfloat16).to(memory_format=torch.channels_last)
    with torch.cpu.amp.autocast():
        images = images.to(torch.bfloat16).to(memory_format=torch.channels_last)
        output=model(images)
        print(output)

def bf16_jit_stock_pytorch(model, images, args):
    model = model.to(torch.bfloat16).to(memory_format=torch.channels_last)
    with torch.no_grad():
        with torch.cpu.amp.autocast():
            model = torch.jit.trace(model, torch.randn(args.batch_size, 3, args.height, args.width))
        model = torch.jit.freeze(model)
        with torch.cpu.amp.autocast():
            images = images.to(torch.bfloat16).to(memory_format=torch.channels_last)
            output=model(images)
            print(output)

def fp32_imperative_stock_pytorch_ipex(model, images):
    import intel_pytorch_extension as ipex
    model = model.to(memory_format=torch.channels_last)
    model = ipex.optimize(model, dtype=torch.float32, level="O0")
    images = images.to(memory_format=torch.channels_last)
    output=model(images)
    print(output)

def fp32_jit_stock_pytorch_ipex(model, images, args):
    import intel_pytorch_extension as ipex
    model = model.to(memory_format=torch.channels_last)
    model = ipex.optimize(model, dtype=torch.float32, level="O0")
    model = torch.jit.trace(model, torch.rand(args.batch_size, 3, args.height, args.width).to(memory_format=torch.channels_last))
    model = torch.jit.freeze(model)
    images = images.contiguous(memory_format=torch.channels_last)
    output=model(images)
    print(output)

def bf16_imperative_stock_pytorch_ipex(model, images):
    import intel_pytorch_extension as ipex
    model = model.to(torch.bfloat16).to(memory_format=torch.channels_last)
    model = ipex.optimize(model, dtype=torch.bfloat16, level="O0")
    conf = ipex.AmpConf(torch.bfloat16)
    with ipex.amp.autocast(enabled=True, configure=conf), torch.no_grad():
        output=model(images)
        print(output)

def bf16_jit_stock_pytorch_ipex(model, images, args):
    import intel_pytorch_extension as ipex
    with torch.no_grad():
        model = model.to(torch.bfloat16).to(memory_format=torch.channels_last)
        model = ipex.optimize(model, dtype=torch.bfloat16, level="O0")
        conf = ipex.AmpConf(torch.bfloat16)
        images = images.to(torch.bfloat16)
        with ipex.amp.autocast(enabled=True, configure=conf), torch.no_grad():
            model = torch.jit.trace(model, torch.rand(args.batch_size, 3, args.height, args.width).to(memory_format=torch.channels_last))
            model = torch.jit.freeze(model)
            output=model(images)
            print(output)

def fp32_imperative_pytorch_ipex(model, images):
    import intel_pytorch_extension as ipex
    model = model.to(ipex.DEVICE)
    images = images.to(ipex.DEVICE)
    output=model(images)
    print(output)

def fp32_jit_pytorch_ipex(model, images, args):
    import intel_pytorch_extension as ipex
    model = model.to(ipex.DEVICE)
    images = images.to(ipex.DEVICE)
    with torch.no_grad():
        model = torch.jit.trace(model, torch.randn(args.batch_size, 3, args.height, args.width).to(ipex.DEVICE))
    output=model(images)
    print(output)

def bf16_imperative_pytorch_ipex(model, images):
    import intel_pytorch_extension as ipex
    ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)
    model = model.to(ipex.DEVICE)
    images = images.to(ipex.DEVICE)
    output=model(images)
    print(output)

def bf16_jit_pytorch_ipex(model, images, args):
    import intel_pytorch_extension as ipex
    ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)
    model = model.to(ipex.DEVICE)
    images = images.to(ipex.DEVICE)
    with torch.no_grad():
        model = torch.jit.trace(model, torch.randn(args.batch_size, 3, args.height, args.width).to(ipex.DEVICE))
    output=model(images)
    print(output)

def get_data(args):
    if args.dummy:
        images = torch.randn(args.batch_size, 3, args.height, args.width)
        return images
    else:
        import sys
        import cv2
        img = cv2.imread(args.imgfile)
        sized = cv2.resize(img, (args.width, args.height))
        img = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)
        img = torch.autograd.Variable(img)
        return img

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    n_classes = args.n_classes
    weightfile = args.weightfile
    
    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    img = get_data(args)
    
    model.eval()
    fp32_imperative_stock_pytorch(model, img)
    fp32_jit_stock_pytorch(model, img, args)
    bf16_imperative_stock_pytorch(model, img)
    bf16_jit_stock_pytorch(model, img, args)
    fp32_imperative_stock_pytorch_ipex(model, img)
    fp32_jit_stock_pytorch_ipex(model, img, args)
    bf16_imperative_stock_pytorch_ipex(model, img)
    bf16_jit_stock_pytorch_ipex(model, img, args)
    fp32_imperative_pytorch_ipex(model, img)
    fp32_jit_pytorch_ipex(model, img, args)
    bf16_imperative_pytorch_ipex(model, img)
    bf16_jit_pytorch_ipex(model, img, args)

