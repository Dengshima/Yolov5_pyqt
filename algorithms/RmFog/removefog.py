import shutil
import os


# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from .val_data import ValData
from .model import GridDehazeNet
from .utils import validation


class RemoveFog():
    def __init__(self, paths):
        #目录自己改一下即可，复制
        # path = './data/test/SOTS/outdoor/clear'  
        # new_path = './data/test/SOTS/outdoor/hazy'
        # txt_path = './data/test/SOTS/outdoor/val_list.txt'
        path = paths[0]
        new_path = paths[1]
        if os.path.exists(new_path):
            shutil.rmtree(new_path)  # delete crop folder
        os.makedirs(new_path)  # make new crop folder
        txt_path = paths[2]
        self.val_data_dir = paths[3]
        self.root_path = paths[4]
        with open(txt_path, 'w') as fp:
            for file in os.listdir(path):
                file_new = os.path.splitext(file)[0] + '_1' + os.path.splitext(file)[1]
                print('file_new: ', file_new)
                full_file = os.path.join(path, file)
                new_full_file = os.path.join(new_path, file_new)
                shutil.copy(full_file, new_full_file)
                fp.write(file_new + '\n')

    def hazy(self):
        # --- Parse hyper-parameters  --- #
        parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
        parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
        parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
        parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
        parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
        parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
        parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
        parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='outdoor', type=str)
        args = parser.parse_args()

        network_height = args.network_height
        network_width = args.network_width
        num_dense_layer = args.num_dense_layer
        growth_rate = args.growth_rate
        lambda_loss = args.lambda_loss
        val_batch_size = args.val_batch_size
        category = args.category

        print('--- Hyper-parameters for testing ---')
        print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
            .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

        # --- Set category-specific hyper-parameters  --- #
        # if category == 'indoor':
        #     val_data_dir = './data/test/SOTS/indoor/'
        # elif category == 'outdoor':
        #     val_data_dir = './data/test/SOTS/outdoor/'
        # else:
        #     raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')
        val_data_dir = self.val_data_dir


        # --- Gpu device --- #
        device_ids = [Id for Id in range(torch.cuda.device_count())]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # --- Validation data loader --- #
        val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


        # --- Define the network --- #
        net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


        # --- Multi-GPU --- #
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)


        # --- Load the network weight --- #
        net.load_state_dict(torch.load(self.root_path + '{}_haze_best_{}_{}'.format(category, network_height, network_width), \
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))


        # --- Use the evaluation model in testing --- #
        net.eval()
        print('--- Testing starts! ---')
        start_time = time.time()
        val_psnr, val_ssim = validation(net, val_data_loader, device, category, save_tag=True)
        end_time = time.time() - start_time
        print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
        print('validation time is {0:.4f}'.format(end_time))
