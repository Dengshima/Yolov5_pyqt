import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from . import dataloader, dce_model
# import dce_model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import shutil
import yaml


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)

    if torch.cuda.is_available():
        data_lowlight = data_lowlight.cuda().unsqueeze(0)
        DCE_net = dce_model.enhance_net_nopool().cuda()
    else:
        data_lowlight = data_lowlight.unsqueeze(0)
        DCE_net = dce_model.enhance_net_nopool()
    # DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DCE_net.load_state_dict(torch.load('algorithms/ZeroDCE/snapshots/Epoch99.pth', map_location=device))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    # image_path = image_path.replace('test_data','result')
    # result_path = image_path
    file_name = image_path.split('/')[-1]

    with open('config.yaml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    result_path = config['lowlight']
    if os.path.exists(result_path):
        shutil.rmtree(result_path)  # delete crop folder
    os.makedirs(result_path)  # make new crop folder

    result_path = config['lowlight'] + '/' + file_name
    print('Enlighten result: ', result_path)
    # if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
    #  os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, result_path)
    shutil.copy(result_path, config['temp'])
