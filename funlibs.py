# Author: Deng Shi Ma
# Date: 2021-04-06
import os
import shutil
import yaml
import random
import cv2
import numpy as np

from PIL import Image

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QTableWidgetItem


def init_dir(dir_path):
    '''
    清空目录中的内容
    输入dir_path: str类型，需要清空的目录
    '''
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # delete crop folder
    os.makedirs(dir_path)  # make new crop folder


def print_txt(tableWidget, targets, result):
    '''
    读取检测结果文件，输出到表格控件中
    输入：tableWidget: qt的tableWidget控件，用于显示结果列表
    targets: 目标类别字典
    result: 检测结果的图片路径（列表）
    '''
    tableWidget.setRowCount(0)
    # image_bboxs 存放每张图片的检测框，元素单位是一张图片的检测框集合
    image_bboxs = []
    for image in result:
        txt_name = str(image.split('.')[:-1][0]) + '.txt'
        if not os.path.exists(txt_name):
            continue
        f = open(txt_name)
        bboxs = f.read().splitlines()[1:]
        image_bboxs.append(bboxs)
    # row_datas 存放每个框的类别和置信度
    row_datas = []
    # bboxs 是一张图片的所有检测框，元素单位是一个框
    for bboxs in image_bboxs:
        for bbox in bboxs:
            bbox = bbox.split(' ')
            row_datas.append([targets[bbox[0]], bbox[1]])
    # 按照舰船目标种类排序再显示
    row_datas.sort()
    for row_data in row_datas:
        addTableRow(tableWidget, row_data)


def addTableRow(table, row_data):
    '''
    为tableWidget添加一行数据
    输入：
    table: tableWidget控件
    row_data: 需要显示的数据，一行
    '''
    row = table.rowCount()
    table.setRowCount(row+1)
    col = 0
    for item in row_data:
        cell = QTableWidgetItem(str(item))
        table.setItem(row, col, cell)
        col += 1


def plotline(origin, colums, rows, h=0, v=0):
    save_name = os.path.basename(origin)
    with open('config.yaml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    init_dir(config['croped'])
    save_name = config['croped'] + '/' + save_name

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(colums * rows)]
    # red = (0, 0, 255)
    # origin = Image.open(origin)
    origin = cv2.imread(origin)
    # width, height = origin.size
    height = origin.shape[0]
    width = origin.shape[1]
    # 每张图片的宽度
    stride_r = width / rows
    # 图片高度
    stride_c = height / colums
    # 存放切割结果的图片路径
    for j in range(colums):
        for i in range(rows):
            box = [i*stride_r, j*stride_c, (i+1)*stride_r, (j+1)*stride_c]
            # 向右下角加重叠部分
            if j != colums-1 and i != rows-1:
                box[2] += h*stride_r
                box[3] += v*stride_c
            # 右下角的一块，往左上加重叠部分
            elif j == colums-1 and i == rows-1:
                box[0] -= h*stride_r
                box[1] -= v*stride_c
            # 行或者列的边际，往图像内部加重叠
            elif j == colums-1:
                box[1] -= v*stride_c
                box[2] += h*stride_r
            else:
                box[0] -= h*stride_r
                box[3] += v*stride_c
            box = tuple(box)
            color = colors[j * colums + i]
            cv2.rectangle(origin, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 3)
    cv2.imwrite(save_name, origin)
    return save_name


def cropimage_overlap(origin, colums, rows, save_path, h=0, v=0):
    '''
    带重叠率切割图片的函数，
    输入：
    origin: str, 原图，图片路径
    colums: int 切割后行数
    rows: int 切割后列数
    save_path: str, 保存切割后的图片的路径
    h: float, 水平切割重叠率
    v: float, 垂直切割重叠率
    返回：
    result: list[image_result, new_imageName], 列表形式
    第一个参数为切割后的实际图片路径列表，第二个为显示的图片路径（图上画切割线的形式）
    '''
    save_name = plotline(origin, colums, rows, h=h, v=v)
    filetype = origin.split('.')[-1]
    origin = Image.open(origin)
    width, height = origin.size
    # 每张图片的宽度
    stride_r = width / rows
    # 图片高度
    stride_c = height / colums
    # 存放切割结果的图片路径
    init_dir(save_path)
    images = []
    for j in range(colums):
        for i in range(rows):
            box = [i*stride_r, j*stride_c, (i+1)*stride_r, (j+1)*stride_c]
            # 向右下角加重叠部分
            if j != colums-1 and i != rows-1:
                box[2] += h*stride_r
                box[3] += v*stride_c
            # 右下角的一块，往左上加重叠部分
            elif j == colums-1 and i == rows-1:
                box[0] -= h*stride_r
                box[1] -= v*stride_c
            # 行或者列的边际，往图像内部加重叠
            elif j == colums-1:
                box[1] -= v*stride_c
                box[2] += h*stride_r
            else:
                box[0] -= h*stride_r
                box[3] += v*stride_c
            box = tuple(box)
            imgname = save_path + '/' + str(j) + str(i) + '.' + filetype
            images.append(imgname)
            # 切图并保存文件
            origin.crop(box).save(imgname)
    # return images
    return [images, save_name]


def concat_image(image_names, colums, rows, overlap_h=0, overlap_v=0):
    '''
    拼接图片的函数

    输入：

    image_names: 图片路径，列表
    colums: 行数，int
    rows: 列数, int
    overlap_h: 水平重叠率
    overlap_v: 垂直重叠率
    返回：
    img_result: 图片拼接结果，ndarray格式
    '''
    # 切割后的图片大小
    h = cv2.imread(image_names[0]).shape[0]
    w = cv2.imread(image_names[0]).shape[1]

    # 需要转换成无重叠的大小：
    # 计算方式：(1 + overlap_h) * stride = w
    stride_r = int(w / (1 + overlap_h))
    stride_c = int(h / (1 + overlap_v))
    img_r = []
    for j in range(colums):
        img_c = []
        for i in range(rows):
            img = cv2.imread(image_names[j*rows+i])
            # *****************根据重叠率计算重叠部分，并去除*********************
            # 向右下角重叠部分
            box = np.copy(img)
            if j != colums-1 and i != rows-1:
                box, box2 = np.split(box, [stride_c])
                box, box2 = np.split(box, [stride_r], axis=1)
            # 右下角的一块，往左上重叠部分
            elif j == colums-1 and i == rows-1:
                box2, box = np.split(box, [int(overlap_v*stride_c)])
                box2, box = np.split(box, [int(overlap_h*stride_r)], axis=1)
            # 行或者列的边际，往图像内部重叠
            elif j == colums-1:
                box2, box = np.split(box, [int(overlap_v*stride_c)])
                box, box2 = np.split(box, [stride_r], axis=1)
            else:
                box, box2 = np.split(box, [stride_c])
                box2, box = np.split(box, [int(overlap_h*stride_r)], axis=1)
            box = cv2.resize(box, (stride_r, stride_c), interpolation=cv2.INTER_NEAREST)
            img = np.copy(box)

            img_c.append(img)
        img_r.append(cv2.hconcat(img_c))
    img_result = cv2.vconcat(img_r)
    return img_result


def showImages(gridLayout, colums, rows, imgnamelist):
    '''
    显示图片函数
    给该函数传入需要显示的图像路径（以列表形式），即可在窗口右边模块显示

    输入：
    gridLayout: widget对应的gridLayout
    colums: 行数
    rows: 列数
    imgnamelist: 需要展示的图片路径列表
    '''
    n = len(imgnamelist)
    # 先清除原窗口已显示的图像
    for i in range(gridLayout.count()):
        gridLayout.itemAt(i).widget().deleteLater()

    labels = []

    # if n != 0:
    #    width = widget.width() / rows - 20
    #    height = widget.height() / colums - 20
    for imgName in imgnamelist:

        image = cv2.imread(imgName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blue = cv2.split(image)[0]
        # 参数依次为：图像、宽、高、每一行的字节数、图像格式彩色图像一般为Format_RGB888
        image2 = QImage(image, image.shape[1], image.shape[0], blue.shape[1]*3,
                        QImage.Format_RGB888)
        image = QPixmap(image2)  # .scaled(width, height)  # QImage类型图像放入QPixmap

        # image = QtGui.QPixmap(imgName).scaled(width, height)
        # 根据图片动态生成label
        label = QLabel()
        label.setPixmap(image)
        label.setScaledContents(True)
        labels.append(label)

    # 根据是否被切分显示图片结果
    if n == 0:
        return
    elif n == 1:
        gridLayout.addWidget(labels[0], 0, 0, 1, 1)
    else:
        for j in range(colums):
            for i in range(rows):
                gridLayout.addWidget(labels[j*rows+i], j, i, 1, 1)
