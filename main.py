# Author: Deng Shi Ma
# Date: 2021.03.25
import sys
import os
import queue
import cv2
import yaml
import shutil
import time
from multiprocessing import Process, Queue
# from tqdm import tqdm
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QThread
from ui.mainUI import Ui_MainWindow
from ui.trainParasMain import TrainWindow
import qdarkstyle

from funlibs import init_dir, print_txt, cropimage_overlap, concat_image, showImages
from yoloThreads import EnlightenWork, DetectThread, OutputThread, FogThread
from algorithms.Yolov5 import Yolov5_train


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 读取配置文件
        with open('config.yaml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # 设置按钮响应函数**************************************
        # 将按钮响应函数整合到一个函数，方便修改管理，修改option字典对应函数即可
        # 对应规则即，根据对应按钮或者菜单选项的名字，查字典，获取对应函数
        self.treeWidget.itemClicked.connect(lambda:
                                            self.onclick(self.treeWidget.currentItem().text(0)))
        self.options = {'打开文件': 'openfile', '关闭文件': 'reset', '大图切割': 'cropimage',
                        '亮度增强': 'enlighten', '图像去雾': 'ridfog', '小图合并': 'concat',
                        '选择模型': 'choosemodel', '参数设置': 'selectdata', '开始训练': 'train',
                        '终止训练': 'stoptrain', '执行检测': 'detection', '模型剪枝': 'lightweight',
                        '重置默认': 'default'}
        self.action.triggered.connect(self.exitprogram)  # 退出程序

        # 当需要传入参数时，要使用lambda表达式函数
        self.pushButton.clicked.connect(lambda: self.onclick('大图切割'))
        self.pushButton_2.clicked.connect(lambda: self.onclick('执行检测'))
        self.pushButton_3.clicked.connect(lambda: self.textBrowser.clear())
        self.pushButton_4.clicked.connect(self.changetrain)
        self.pushButton_5.clicked.connect(self.showtrain)

        # 模型选择菜单*********************************************
        self.models_dict = {'深度学习目标检测识别模型(RGB)': 'weights/RGBbest.pt',
                            '基于样本迁移的目标检测识别模型(RGB)': 'weights/RGBbest.pt',
                            '基于特征迁移的目标检测识别模型(PAN)': 'weights/PANbest.pt',
                            '基于模型迁移的目标检测识别模型(IR)': 'weights/IRbest.pt'}

        # 按键选择对应模型权重
        self.comboBox.currentIndexChanged.connect(
            lambda: self.changemodel(self.models_dict[self.comboBox.currentText()])
        )

        # 图片切割规模
        self.rows = 2   # rows
        self.colums = 2   # columns

        # 主线程， 判断任务队列是否存在待执行的任务
        self.th = OutputThread(data=self)
        # 文本重定向至GUI控件
        self.th.signalForText.connect(self.outputWritten)
        self.th.judge.connect(self.judge_thread)
        # 将控制台文本重定向至GUI控件中
        # sys.stdout = self.th
        # sys.stderr = self.th
        self.th.start()

        # 训练参数配置子窗口
        self.train_win = TrainWindow()

        # 窗口，变量重置
        self.reset(start=True)

        # 线程队列
        self.qmain = queue.Queue()

        # 获取训练输出消息使用的队列
        self.train_output_q = Queue()

        # 计算线程执行时间的队列
        self.time_queue = Queue()

        # 当前任务清空
        self.current_task = None

    def judge_thread(self):
        '''
        主线程后台一直在执行的函数
        判断任务队列中是否存在待执行的任务
        同时只能有一个任务在执行（训练程序单独运行）
        并且获取训练程序中的输出，展示进度条等
        '''
        # 将训练进程中的输出，显示到GUI控件中
        if not self.train_output_q.empty():
            s = self.train_output_q.get()
            # 开始训练之后，返回训练进度条信息，和当前epoch
            if isinstance(s, dict):
                batchs = s['batchs']
                i = s['i']
                epoch = s['epoch']
                self.train_win.progressBar.setMaximum(batchs)
                self.train_win.progressBar.setValue(i)
                self.progressBar.setValue(epoch)
            # 返回一个列表的时候，代表已经开始训练，具体见训练函数中的返回值，
            # 此处需要不断更新当前训练的数据，所以需要删除之前的最后一行，
            # 再添加新一行
            elif len(s) == 2:
                # 设置光标，删除最后一行
                storeCursorPos = self.train_win.textBrowser.textCursor()
                self.train_win.textBrowser.moveCursor(QtGui.QTextCursor.End,
                                                      QtGui.QTextCursor.MoveAnchor)
                self.train_win.textBrowser.moveCursor(QtGui.QTextCursor.StartOfLine,
                                                      QtGui.QTextCursor.MoveAnchor)
                self.train_win.textBrowser.moveCursor(QtGui.QTextCursor.End,
                                                      QtGui.QTextCursor.KeepAnchor)
                self.train_win.textBrowser.textCursor().removeSelectedText()
                self.train_win.textBrowser.textCursor().deletePreviousChar()
                self.train_win.textBrowser.setTextCursor(storeCursorPos)
                # 添加新一行，达到刷新显示数据的目的
                s = s[0]
                self.train_win.textBrowser.append(s)
            else:
                self.train_win.textBrowser.append(s)
        if self.time_queue.qsize() == 3:
            print('完成')
            string = self.time_queue.get()
            start = self.time_queue.get()
            end = self.time_queue.get()
            self.textBrowser.append(string + ('%.4f') % (end-start) + 's')
        # 任务队列为空
        if self.qmain.empty():
            return
        # 执行队列中的任务
        else:
            self.current_task = self.qmain.get()
            self.current_task.start()

    def changemodel(self, model_weight):
        '''
        修改后面检测使用的权重文件
        :param: model_weight: 检测使用的权重文件, str, 路径
        '''
        self.model_weight = model_weight
        print(self.model_weight)

    def choosemodel(self):
        '''
        选择已训练好的权重文件
        '''
        selected_filter = "Models (*.pt);;All Files(*)"
        FileName, FileType = QFileDialog.getOpenFileName(self, "打开图片", "", selected_filter)
        if FileName != '':
            self.model_weight = FileName.replace(os.getcwd()+'/', '', 1)
            # print(self.model_weight)

    def reset(self, start=False):
        '''
        重置函数

        重置变量，初始化, 输入：

        :param: start: 决定是否重置重叠率，模型等变量参数, bool
                    True: 重置重叠率，模型等所有参数
                    False：不重置上述参数，仅关闭正在显示的图片
        '''
        # 左侧树展开
        self.treeWidget.expandToDepth(0)

        # 结果展示列表重置
        self.tableWidget.setRowCount(0)

        # 共享变量的重置
        self.images = []
        self.sourceimage = ''
        self.result = []

        # 控制台日志清空
        self.textBrowser.clear()
        # self.textBrowser_2.clear()

        # 清空目录
        init_dir(self.config['output_dir'])
        init_dir(self.config['source'])
        init_dir(self.config['temp'])
        init_dir(self.config['result'])

        # 关闭文件不执行，重置会执行
        if start:
            # 切割重叠率重置
            self.doubleSpinBox.setValue(0.0)
            self.doubleSpinBox_2.setValue(0.0)

            # 设置默认模型
            self.comboBox.setCurrentText('深度学习目标检测识别模型(RGB)')
            self.model_weight = self.models_dict['深度学习目标检测识别模型(RGB)']
        else:
            # 对应关闭文件选项
            showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, [])

    def default(self):
        '''
        重置变量，模型等参数
        '''
        self.reset(start=True)

    def exitprogram(self):
        '''
        退出程序
        '''
        try:
            self.train_pro.kill()
            self.train_pro.close()
        except Exception as e:
            print(e)
        exit()

    def outputWritten(self, text):
        '''
        重定向文本输出
        '''
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def onclick(self, option):
        '''
        根据按钮名字，执行对应响应函数
        '''
        # 当前有任务执行，点击不产生反应
        if self.current_task is not None:
            return
        try:
            # 根据按钮名称获取响应函数，并执行对应函数
            getattr(self, self.options[option])()
        except Exception as e:
            print(e)

    def openfile(self):
        '''
        打开一张图片
        '''
        self.tableWidget.setRowCount(0)
        # 清空目录
        # init_dir(self.config['output_dir'])
        # 选择打开选项，图片格式，所有文件，括号中使用匹配符号
        selected_filter = "Images (*.png *.jpg *.JPEG);;All Files(*)"
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", selected_filter)
        # 当未选中图片取消时，直接返回
        if len(imgName) == 0:
            return
        else:
            self.reset(start=False)
            # 获取图片的相对路径
            imgName = imgName.replace(os.getcwd()+'/', '', 1)
            # 清空source路径，并将图片移动至source中
            init_dir(self.config['source'])
            init_dir(self.config['temp'])
            init_dir(self.config['result'])
            self.sourceimage = shutil.copy(imgName, self.config['source'])
            image = shutil.copy(self.sourceimage, self.config['temp'])

            self.images = [image]
            showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, self.images)

    def cropimage(self):
        '''
        切图函数
        无需传参，根据self.images列表进行切割
        无返回值，切割结果赋值给self.images（列表形式），并显示
        '''
        # 如果当前图片不是一张，则为误操作
        if len(self.images) != 1:
            return
        # 获取设置的重叠率
        overlap_h = self.doubleSpinBox.value()
        overlap_v = self.doubleSpinBox_2.value()
        self.images = cropimage_overlap(self.images[0], self.colums,
                                        self.rows, self.config['temp'],
                                        h=overlap_h, v=overlap_v)
        # self.showImages(self.images)
        showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, self.images)
        self.tableWidget.setRowCount(0)

    def enlighten(self):
        '''
        self.images(增强前)  ->  self.images（增强后）
        showImages 显示
        '''
        if len(self.images) == 0:
            return
        # 后台线程执行亮度增强函数
        # 新建一个线程
        self.lighten_th = QThread()
        self.lighten_wk = EnlightenWork(self.images)
        self.lighten_wk.moveToThread(self.lighten_th)

        self.lighten_th.started.connect(self.lighten_wk.run)
        self.lighten_wk.finished.connect(self.lighten_th.quit)
        self.lighten_wk.finished.connect(self.lighten_wk.deleteLater)
        self.lighten_th.finished.connect(self.lighten_th.deleteLater)

        self.lighten_wk.finished.connect(self.enlighten_finished)

        self.qmain.put(self.lighten_th)
        # 当前时间入队列
        self.time_queue.put('亮度增强耗时: ')
        self.time_queue.put(time.time())

    def enlighten_finished(self):
        '''
        当亮度增强算法运行结束后，执行的函数
        显示增强后的图片
        '''
        lowlight_result = self.config['temp']
        image_names = [(lowlight_result + '/' + name) for name in os.listdir(lowlight_result)]
        image_names.sort()
        self.images = image_names
        print('enlighten result:', self.images)
        showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, self.images)
        self.tableWidget.setRowCount(0)
        self.current_task = None
        self.time_queue.put(time.time())

    def ridfog(self):
        '''
        对应目录解释在__init__.py中
        self.images -> self.images
        增强前  ->  增强后
        '''
        paths = self.config['fog_dir']
        target = paths[0]
        out_result = self.config['fog_outdoor']
        init_dir(target)
        init_dir(out_result)
        if len(self.images) == 0:
            return
        print('going to remove fog:', self.images)
        for file in self.images:
            try:
                shutil.copy(file, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except Exception:
                print("Unexpected error:", sys.exc_info())

        self.ridfog_th = FogThread(paths)
        self.ridfog_th.finished.connect(self.ridfog_result)
        self.qmain.put(self.ridfog_th)
        self.time_queue.put('去雾耗时: ')
        self.time_queue.put(time.time())

        # rf = removefog.RemoveFog(paths)
        # rf.hazy()
        # self.ridfog_result()

    def ridfog_result(self):
        out_result = self.config['fog_outdoor']
        image_names = [(out_result + '/' + name) for name in os.listdir(out_result)]
        image_names.sort()
        init_dir(self.config['temp'])
        results = []
        for image in image_names:
            results.append(shutil.copy(image, self.config['temp']))
        # self.images = image_names
        self.images = results
        # self.showImages(self.images)
        print('remove fog result:', self.images)
        showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, self.images)
        self.tableWidget.setRowCount(0)
        self.time_queue.put(time.time())
        self.current_task = None

    def concat(self):
        '''
        图片拼接函数

        图片拼接有两种情况，
        1. 原图未检测，切割后直接拼接
        2. 原图切割后，对检测结果进行拼接
        目前的检测后拼接方案为“假拼接”，实际并没有对检测后的小图拼接，
        而是直接使用原大图进行检测后再拼接
        '''
        # 检测结果图
        n = len(self.result)
        # 待检测原图
        m = len(self.images)
        # 待检测的小图均已检测，拼接检测结果
        if m > 1 and n > 1:
            concat_path = self.config['result']
            # 使用拼接原切割图，再使用大图检测的“假拼接方式”
            # concat_path = self.config['temp']
        # 小图未检测，拼接回原图
        elif m > 1:
            concat_path = self.config['temp']
        # 图片不需要拼接
        else:
            return
        image_names = []
        for name in os.listdir(concat_path):
            # 排除txt文件
            if name[-4:] != '.txt':
                image_names.append(concat_path + '/' + name)
        # image_names = [(concat_path + '/' + name) for name in os.listdir(concat_path)]
        image_names.sort()

        img_result = concat_image(image_names, self.colums, self.rows,
                                  overlap_h=self.doubleSpinBox.value(),
                                  overlap_v=self.doubleSpinBox_2.value())
        # 清空原来路径，并将新图保存至原来路径
        init_dir(concat_path)
        cv2.imwrite(image_names[0], img_result)
        # 如果是拼接原图，则self.images还是拼接后的图
        # 如果是拼接检测结果，则self.result应改变为拼接结果
        if concat_path == self.config['temp']:
            self.images = [image_names[0]]
        else:
            self.result = [image_names[0]]

        showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, [image_names[0]])

    def detection(self):
        '''
        根据单张图片还是多张图片，
        传入图片名或者路径
        '''
        # start = time.time()

        # 'RGB', 'PAN', 'IR'
        # model_weight = self.models_dict[self.comboBox.currentText()]
        if len(self.images) == 0:
            return
        detect_dir = os.path.dirname(self.images[0])
        self.detect_th = DetectThread([detect_dir, self.model_weight])

        self.detect_th.finished.connect(self.detect_result)
        self.qmain.put(self.detect_th)
        self.time_queue.put('检测耗时: ')
        self.time_queue.put(time.time())

        # end = time.time()
        # cost_time = end - start
        # self.textBrowser_2.append('检测耗时:' + str(cost_time) + 's')

    def detect_result(self, result):
        self.result = result
        init_dir(self.config['result'])
        result0 = []
        for image in self.result:
            result0.append(shutil.copy(image, self.config['result']))
        self.result = result0
        showImages(self.widget_4, self.gridLayout_2, self.colums, self.rows, self.result)
        print_txt(self.tableWidget, self.config['targets'], result)
        self.time_queue.put(time.time())
        self.current_task = None

    def selectdata(self):
        '''
        训练参数设置
        '''
        self.train_win.show()
        self.train_win.exec_()

    def changetrain(self):
        if self.pushButton_4.text() == '开始训练':
            self.train()
        else:
            self.stoptrain()

    def showtrain(self):
        self.train_win.show()
        self.train_win.exec()

    def stoptrain(self):
        try:
            self.train_pro.kill()
            self.train_pro.close()
        except Exception as e:
            print(e)
        self.pushButton_4.setText('开始训练')
        print('stoptrain')

    def train(self):
        '''
        开始训练
        '''
        # 设置界面设置训练参数
        dataset = self.train_win.dataset
        weights = self.train_win.weights
        batchsize = self.train_win.batchsize
        epochs = self.train_win.epochs
        print("", dataset, weights, batchsize, epochs)
        try:
            del self.train_pro
        except Exception as e:
            print(e)
        self.train_pro = Process(target=Yolov5_train.train0, args=(
            'train.yaml', 'models/yolov5s.yaml', '', batchsize, epochs, self.train_output_q,
        ))
        self.train_win.textBrowser.clear()
        self.train_pro.start()
        self.train_win.show()
        self.pushButton_4.setText('停止训练')
        self.progressBar.setMaximum(epochs - 1)
        self.train_win.label_3.setText('epoch:0/' + str(epochs-1))

    def lightweight(self):
        '''
        模型剪枝函数，，，害，假的
        '''
        self.model_weight = 'weights/Fast.pt'

    def closeEvent(self, event):
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    # 使用qdarkstyle风格
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    myWin.show()
    sys.exit(app.exec_())
