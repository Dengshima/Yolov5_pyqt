from PyQt5.QtWidgets import QDialog, QFileDialog
from trainParasUI import Ui_Dialog
import yaml
import os


class TrainWindow(QDialog, Ui_Dialog):
    '''
    配置训练参数的小弹窗
    '''
    def __init__(self, parent=None):
        super(TrainWindow, self).__init__(parent)
        self.setupUi(self)
        self.options = {'选择数据集路径': 'select_data', '预训练模型': 'pretrained',
                        'confirm': 'confirm', 'cancel': 'cancel'}
        # 按钮绑定函数
        self.buttonBox.accepted.connect(lambda: self.onclick('confirm'))
        self.buttonBox.rejected.connect(lambda: self.onclick('cancel'))
        self.pushButton.clicked.connect(lambda: self.onclick('选择数据集路径'))
        self.pushButton_2.clicked.connect(lambda: self.onclick('预训练模型'))

        # 默认设置
        self.dataset = 'shipdata'
        self.modelcfg = 'Yolov5train/models/yolov5s.yaml'
        self.weights = 'weights/yolov5s.pt'
        self.datasetconfig = 'train.yaml'
        self.batchsize = 2
        self.epochs = 100
        self.settings()

    def settings(self):
        '''
        根据预定配置，设置默认参数
        '''
        with open('config.yaml', 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.lineEdit.setText(self.dataset)
        self.lineEdit_2.setText(self.weights)
        self.spinBox.setProperty("value", self.batchsize)
        self.spinBox_2.setProperty("value", self.epochs)
        # 数值框改变，触发函数
        self.spinBox.valueChanged.connect(self.changevalue)
        self.spinBox_2.valueChanged.connect(self.changevalue)

    def changevalue(self):
        self.batchsize = self.spinBox.value()
        self.epochs = self.spinBox_2.value()

    def onclick(self, option):
        '''
        根据按钮名字，执行对应响应函数
        '''
        try:
            # 根据按钮名称获取响应函数，并执行对应函数
            getattr(self, self.options[option])()
        except Exception as e:
            print(e)

    def select_data(self):
        '''
        训练参数配置
        '''
        select = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        # print(os.getcwd())
        self.dataset = select.replace(os.getcwd()+'/', '', 1)
        self.lineEdit.setText(self.dataset)

    def pretrained(self):
        '''
        选择预训练模型
        '''
        selected_filter = "Checkpoint (*.pt);;All Files(*)"
        FileName, FileType = QFileDialog.getOpenFileName(self, "打开权重文件", "", selected_filter)
        self.weights = FileName.replace(os.getcwd()+'/', '', 1)
        self.lineEdit_2.setText(self.weights)

    def confirm(self):
        '''
        确认配置
        '''
        print('confirm')
        self.hide()

    def cancel(self):
        '''
        取消设置
        '''
        print('cancel')
        self.hide()
