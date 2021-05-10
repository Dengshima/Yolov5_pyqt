import time

from PyQt5.QtCore import pyqtSignal, QObject, QThread

from algorithms.Yolov5 import Yolov5_detect
from algorithms.ZeroDCE import lowlight_test
from algorithms.RmFog import removefog

'''
关于PyQt的多线程(或者多进程)方案：
一. 多线程
    Python 的多线程无法完全利用CPU资源，仅会使用当前核心的CPU开启多线程，
所以无法最大化CPU计算资源，适用于并不是特别复杂的任务，只是想要实现
多个简单的代码逻辑同时运行，可以使用多线程，因为线程之间交互更方便。
    Python自带thread，和PyQt的Qthread均可实现多线程，Qthread与Qt的GUI之间
交换信息更方便
1. QThread多线程
Qthread 多线程有两种实现方式：
(1) 如DetectThread的使用方式，直接继承QThread类，重写run函数逻辑，最简单，但是***********
QT作者批评这种方式，不建议这么做，防止以后出问题，可以使用(2)
(2) 自定义QObject类，在QObject中写函数逻辑，再使用moveToThread（见EnlightWork的使用）
比（1）多几行代码，略微麻烦一点，但以后发生变化，可能兼容性更高

二. 多进程
    由于Python多线程的特殊性，有时候多线程并不能满足性能要求，可以使用多进程，
训练代码使用的多进程方案，后台执行训练代码，多进程数据通信使用Queue对象
'''


class OutputThread(QThread):
    '''
    后台判断线程
    '''
    signalForText = pyqtSignal(str)
    judge = pyqtSignal()

    def __init__(self, data=None, parent=None):
        super(OutputThread, self).__init__(parent)
        self.data = data

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):
        print('主程序启动')
        while True:
            time.sleep(0.05)
            self.judge.emit()


class EnlightenWork(QObject):
    '''
    亮度增强线程
    '''
    # QT触发信号
    finished = pyqtSignal()

    def __init__(self, data=None, parent=None):
        super(EnlightenWork, self).__init__(parent)
        self.data = data

    def run(self):
        print(self.data)
        images = self.data
        for image in images:
            lowlight_test.lowlight(image)
        # 当运行结束，触发该信号
        self.finished.emit()


class DetectThread(QThread):
    '''
    检测代码
    '''
    finished = pyqtSignal(list)

    def __init__(self, data=None, parent=None):
        super(DetectThread, self).__init__(parent)
        self.data = data

    def run(self):
        source = self.data[0]
        weights = self.data[1]
        if len(self.data) == 3:
            result = Yolov5_detect.detect(source, weights, output=self.data[2])
        else:
            result = Yolov5_detect.detect(source, weights)
        self.finished.emit(result)


class FogThread(QThread):
    '''
    去雾线程类
    '''
    finished = pyqtSignal()

    def __init__(self, data=None, parent=None):
        super(FogThread, self).__init__(parent)
        self.data = data

    def run(self):
        paths = self.data
        rf = removefog.RemoveFog(paths)
        rf.hazy()
        self.finished.emit()
