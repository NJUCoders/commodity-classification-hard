import csv
import argparse
import os
import sys

import numpy as np
import torch
import torch.cuda
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from my.yolov3.easy.net.load_net import load_net
from PIL import Image

image_size = (96, 96)


test_transformations = transforms.Compose([
    transforms.ToTensor()
])


def load_trained_net(model_path):
    print("Begin to load pre-trained net ... ", end="")
    net = load_net("resnet152")
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print("Finished.")
    return net


def predict(net, ims: list):
    # Define transformations for the image
    transformation = test_transformations

    images_tensor_list = []
    for im in ims:
        w = max(im.size)  # 正方形的宽度
        im = im.crop((0, 0, w, w)).resize(image_size)  # 补成正方形再压缩
        image = np.asarray(im)
        image_tensor = transformation(image)
        images_tensor_list.append(image_tensor)

    images_tensor = torch.stack(images_tensor_list)

    if torch.cuda.is_available():
        images_tensor.cuda()

    # 将输入变为变量
    input = Variable(images_tensor)

    # 预测图像的类
    output = net(input)
    index = output.data.numpy().argmax(axis=1)
    return index + 1  # [0, C-1] -> [1, C]

if __name__ == '__main__':
    net = load_trained_net("model/model-87-8.477896466274615e-05.pth")

    image_paths = ["../data/images/0a0bf7bc-e0d7-4f20-abec-039136663d85.jpg",
            "../data/images/0a0c27d7-2e2a-4817-a715-8182cf07ec9b.jpg",
            "../data/images/0a00c2a3-a498-452a-ba88-6b9ef514e201.jpg",
            "../data/images/0a1a5d35-1b30-43ff-87bc-9acdab1567c1.jpg"]

    ims = []

    for image_path in image_paths:
        im = Image.open(image_path)
        ims.append(im)

    results = predict(net, ims)
    print(results)
