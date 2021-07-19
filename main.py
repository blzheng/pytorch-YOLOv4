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
from cfg import Cfg
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tool.utils import *

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='PyTorch Yolov4 Training')
    parser.add_argument('-N', '--n_classes', default=80, type=int, metavar='n_classes',
                        help='num classes')
    parser.add_argument('-w', '--weightfile', type=str, default='./yolov4.pth',
                        help='weight file')
    parser.add_argument('-i', '--imgfile', type=str, default='./data/dog.jpg',
                        help='image file')
    parser.add_argument('--height', default=416, type=int, help='height')
    parser.add_argument('--width', default=416, type=int, help='width')
    parser.add_argument('-n', '--namesfile', type=str, help='names file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--xpu', action='store_true', default=False, 
                        help='use IPEX XPU')
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
    parser.add_argument('--warmup', default=10, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('--max_iter', default=30, type=int, 
                        help='max iterations to run')
    parser.add_argument('--train', action='store_true',
                        help='do train')
    parser.add_argument('--evaluate', action='store_true',
                        help='do evaluate')
    parser.add_argument('--profile', action='store_true',
                        help='do profile')
    parser.add_argument('--val_label_path', dest='val_label', type=str, default='./data/val.txt', help="validate label path")
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 64), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-dir', '--data-dir', type=str, default='/lustre/dataset/COCO2017/val2017/',
                        help='dataset dir', dest='dataset_dir')
    args = vars(parser.parse_args())

    cfg.update(args)

    return edict(cfg)

def get_result(images, targets, output):
    targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
    res = {}
    for img, target, boxes, confs in zip(images, targets, output[0], output[1]):
        img_height, img_width = img.shape[-2], img.shape[-1]
        boxes = boxes.squeeze(2).cpu().detach().numpy()
        boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
        boxes[...,0] = boxes[...,0]*img_width
        boxes[...,1] = boxes[...,1]*img_height
        boxes[...,2] = boxes[...,2]*img_width
        boxes[...,3] = boxes[...,3]*img_height
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        confs = confs.cpu().detach().numpy()
        labels = np.argmax(confs, axis=1).flatten()
        labels = torch.as_tensor(labels, dtype=torch.int64)
        scores = np.max(confs, axis=1).flatten()
        scores = torch.as_tensor(scores, dtype=torch.float32)
        res[target["image_id"].item()] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
    return res

def validation(model, val_loader, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    ap = AverageMeter('AP', ':6.2f')
    ap50 = AverageMeter('AP50', ':6.2f')
    number_iter = len(val_loader)
    if args.calibration:
        number_iter = 100
    progress = ProgressMeter(
        number_iter,
        [batch_time, ap, ap50],
        prefix='Test: ')
    model.eval()
    coco = convert_to_coco_api(val_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')
    if args.ipex:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.warmup:
                    end = time.time()
                images = batch[0]
                images = [[cv2.resize(img, (args.width, args.height))] for img in images]
                images = np.concatenate(images, axis=0)
                images = images.transpose(0, 3, 1, 2)
                images = torch.from_numpy(images).div(255.0)
                images = images.to(device='cpu')
                if args.int8:
                    output = model(images)
                else:
                    images = images.to(memory_format=torch.channels_last)
                    if args.xpu:
                        images = images.to(ipex.DEVICE)
                    if args.bf16 and not args.xpu:
                        images = images.to(torch.bfloat16)
                        with ipex.amp.autocast(enabled=True):
                            if args.profile:
                                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                                    with record_function("model_inference"):
                                        output = model(images)
                            else:
                                output = model(images)
                    else:
                        if args.profile:
                            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                                with record_function("model_inference"):
                                    output = model(images)
                        else:
                            output = model(images)
                if i >= args.warmup:
                    batch_time.update(time.time() - end)
                if args.bf16:
                    output = [o.to(torch.float32) for o in output]
                targets = batch[1]
                res = get_result(images, targets, output)
                coco_evaluator.update(res)
                # if i % args.print_freq == 0:
                #     coco_evaluator.synchronize_between_processes()
                #     coco_evaluator.accumulate()
                #     coco_evaluator.summarize()
                #     stats = coco_evaluator.coco_eval['bbox'].stats
                #     ap.update(stats[0], images.size(0))
                #     ap50.update(stats[1], images.size(0))
                #     progress.display(i)
                #     coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')
                if i >= args.max_iter:
                    break
    else:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.warmup:
                    end = time.time()
                images = batch[0]
                images = [[cv2.resize(img, (args.width, args.height))] for img in images]
                images = np.concatenate(images, axis=0)
                images = images.transpose(0, 3, 1, 2)
                images = torch.from_numpy(images).div(255.0)
                images = images.to(device='cpu')
                if args.int8:
                    output = model(images)
                else:
                    images = images.to(memory_format=torch.channels_last)
                    if args.bf16:
                        with torch.cpu.amp.autocast():
                            images = images.to(torch.bfloat16)
                            if args.profile:
                                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                                    with record_function("model_inference"):
                                        output = model(images)
                            else:
                                output = model(images)
                    else:
                        if args.profile:
                            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                                with record_function("model_inference"):
                                    output = model(images)
                        else:
                            output = model(images)
                if i >= args.warmup:
                    batch_time.update(time.time() - end)
                if args.bf16:
                    output = [o.to(torch.float32) for o in output]
                targets = batch[1]
                res = get_result(images, targets, output)
                coco_evaluator.update(res)
                # if i % args.print_freq == 0:
                #     coco_evaluator.synchronize_between_processes()
                #     coco_evaluator.accumulate()
                #     coco_evaluator.summarize()
                #     stats = coco_evaluator.coco_eval['bbox'].stats
                #     ap.update(stats[0], images.size(0))
                #     ap50.update(stats[1], images.size(0))
                #     progress.display(i)
                #     coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')
                if i >= args.max_iter:
                    break
    if args.profile:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    batch_size = args.batch_size
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    print('inference latency %.3f ms'%latency)
    print("Throughput: {:.3f} fps".format(perf))
    # print("AP: {ap.avg:.3f} ".format(ap=ap))
    # print("AP50: {ap50.avg:.3f} ".format(ap50=ap50))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    args = get_args(**Cfg)
    print(args)
    n_classes = args.n_classes
    weightfile = args.weightfile
    
    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    import sys
    import cv2
    val_dataset = Yolo_dataset(args.val_label, args, train=False)
    n_val = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True, collate_fn=val_collate)
    if args.ipex:
        import intel_pytorch_extension as ipex
    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
    if not args.int8:
        model = model.to(memory_format=torch.channels_last)
    criterion = Yolo_loss(device='cpu', batch=args.batch_size, n_classes=n_classes)
    model.eval()
    if args.evaluate and args.ipex:
        if args.int8:
            print("pending, running int8 evalation step\n")
        elif args.xpu:
            if args.bf16:
                ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)
            model = model.to(ipex.DEVICE)
        else:
            if args.bf16:
                conf = ipex.AmpConf(torch.bfloat16)
                model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
            else:
                conf = ipex.AmpConf(torch.float32)
                model = ipex.optimize(model, dtype=torch.float32, level='O0')
        if args.jit:
            x = torch.randn(args.batch_size, 3, args.height, args.width).to(memory_format=torch.channels_last)
            if args.xpu:
                x = x.to(ipex.DEVICE)
                with torch.no_grad():
                    model = torch.jit.trace(model, x)
            else:
                if args.bf16:
                    x = x.to(torch.bfloat16)
                with ipex.amp.autocast(enabled=True, configure=conf), torch.no_grad():
                    model = torch.jit.trace(model, x)
                model = torch.jit.freeze(model)
        validation(model, val_loader, criterion, args)
    elif args.evaluate:
        if args.bf16:
            model = model.to(torch.bfloat16)
        if args.jit:
            x = torch.randn(args.batch_size, 3, args.height, args.width).to(memory_format=torch.channels_last)
            if args.bf16:
                x = x.to(torch.bfloat16)
                with torch.cpu.amp.autocast(), torch.no_grad():
                    model = torch.jit.trace(model, x)
            else:
                with torch.no_grad():
                    model = torch.jit.trace(model, x)
            model = torch.jit.freeze(model)
        validation(model, val_loader, criterion, args)
