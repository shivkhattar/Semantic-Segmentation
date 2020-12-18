import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from runners.inference_runner import InferenceRunner

color_palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                 [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                 [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                 [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                 [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

seg_classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


def pad_inverse(output, image_shape):
    h, w = image_shape
    return output[:h, :w]


def save_result(fname, pred_mask, classes, palette=None, out=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(classes), 3))
    else:
        palette = np.array(palette)
    img_ori = cv2.imread(fname)
    mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        mask[pred_mask == label, :] = color

    cover = img_ori * 0.5 + mask * 0.5
    cover = cover.astype(np.uint8)

    if out is not None:
        _, fullname = os.path.split(fname)
        fname, _ = os.path.splitext(fullname)
        save_dir = os.path.join(out, fname)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'img.png'), img_ori)
        cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask)
        cv2.imwrite(os.path.join(save_dir, 'cover.png'), cover)


def plot(img, mask, cover):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Semantic Segmentation", y=0.95, fontsize=16)
    ax[0].set_title('image')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].set_title('mask')
    ax[1].imshow(mask)
    ax[2].set_title('cover')
    ax[2].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference a segmentatation model')
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--out', default='./save_result')
    args = parser.parse_args()
    return args


def infer(num_classes, out, inference):
    args = parse_args()
    runner = InferenceRunner(num_classes, out, False, True, inference)
    runner.load_from_checkpoint(args.checkpoint)
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = image.shape[:2]
    mask = np.zeros(image_shape)
    output = runner(image, [mask])
    output = pad_inverse(output, image_shape)
    save_result(args.image, output, classes=seg_classes, palette=color_palette, out=args.out)
