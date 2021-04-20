import shutil
import os

#目录自己改一下即可，复制
path = './data/test/SOTS/outdoor/clear'  
new_path = './data/test/SOTS/outdoor/hazy'
txt_path = './data/test/SOTS/outdoor/val_list.txt'

with open(txt_path, 'w') as fp:
    for file in os.listdir(path):
        file_new = file[:-4] + '_1' + file[-4:]
        full_file = os.path.join(path, file)
        new_full_file = os.path.join(new_path, file_new)
        shutil.copy(full_file, new_full_file)
        fp.write(file_new + '\n')