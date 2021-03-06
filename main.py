# Author: Deng Shi Ma
# Date: 2021.03.25
import sys
import os
import queue
import cv2
import yaml
import shutil
import time
from multiprocessing import Queue, Process
import multiprocessing
# import torch
# from torch.multiprocessing import Process
# from multiprocessing import Process, Queue
# from tqdm import tqdm
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, QEvent
from ui.mainUI import Ui_MainWindow
from ui.trainParasMain import TrainWindow
import qdarkstyle

from funlibs import init_dir, print_txt, cropimage_overlap, concat_image, showImages
from yoloThreads import EnlightenWork, DetectThread, OutputThread, FogThread
from algorithms.Yolov5 import Yolov5_train

import platform


# os compatible for MacOSX
# refer to https://github.com/keras-team/autokeras/issues/368
if platform.system().lower() == "darwin":
    class SharedCounter(object):
        """ A synchronized shared counter.

        The locking done by multiprocessing.Value ensures that only a single
        process or thread may read or write the in-memory ctypes object. However,
        in order to do n += 1, Python performs a read followed by a write, so a
        second process may read the old value before the new one is written by the
        first process. The solution is to use a multiprocessing.Lock to guarantee
        the atomicity of the modifications to Value.

        This class comes almost entirely from Eli Bendersky's blog:
        http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/

        """

        def __init__(self, n = 0):
            self.count = multiprocessing.Value('i', n)

        def increment(self, n = 1):
            """ Increment the counter by n (default = 1) """
            with self.count.get_lock():
                self.count.value += n

        @property
        def value(self):
            """ Return the value of the counter """
            return self.count.value


    class Queue(queue.Queue):
        """ A portable implementation of multiprocessing.Queue.

        Because of multithreading / multiprocessing semantics, Queue.qsize() may
        raise the NotImplementedError exception on Unix platforms like Mac OS X
        where sem_getvalue() is not implemented. This subclass addresses this
        problem by using a synchronized shared counter (initialized to zero) and
        increasing / decreasing its value every time the put() and get() methods
        are called, respectively. This not only prevents NotImplementedError from
        being raised, but also allows us to implement a reliable version of both
        qsize() and empty().

        """

        def __init__(self, *args, **kwargs):
            super(Queue, self).__init__(*args, **kwargs)
            self.size = SharedCounter(0)

        def put(self, *args, **kwargs):
            self.size.increment(1)
            super(Queue, self).put(*args, **kwargs)

        def get(self, *args, **kwargs):
            self.size.increment(-1)
            return super(Queue, self).get(*args, **kwargs)

        def qsize(self):
            """ Reliable implementation of multiprocessing.Queue.qsize() """
            return self.size.value

        def empty(self):
            """ Reliable implementation of multiprocessing.Queue.empty() """
            return not self.qsize()



class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # ??????????????????
        with open('config.yaml', 'r', encoding='UTF-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # ????????????????????????**************************************
        # ????????????????????????????????????????????????????????????????????????option????????????????????????
        # ????????????????????????????????????????????????????????????????????????????????????????????????
        self.treeWidget.itemClicked.connect(lambda:
                                            self.onclick(self.treeWidget.currentItem().text(0)))
        self.options = {'????????????': 'openfile', '????????????': 'reset', '????????????': 'cropimage',
                        '???????????????EnlightenGAN???': 'enlighten', '????????????': 'ridfog', '????????????': 'concat',
                        '????????????': 'choosemodel', '????????????': 'selectdata', '????????????': 'train',
                        '????????????': 'stoptrain', '????????????': 'detection', '????????????': 'lightweight',
                        '????????????': 'default', '????????????': 'detectdir'}
        self.action.triggered.connect(self.exitprogram)  # ????????????

        # ????????????????????????????????????lambda???????????????
        self.pushButton.clicked.connect(lambda: self.onclick('????????????'))
        self.pushButton_2.clicked.connect(lambda: self.onclick('????????????'))
        # ????????????
        self.pushButton_3.clicked.connect(lambda: self.textBrowser.clear())
        # ????????????
        self.pushButton_4.clicked.connect(self.changetrain)
        self.pushButton_5.clicked.connect(self.showtrain)

        # ????????????
        self.pushButton_6.clicked.connect(lambda: self.scrollResize(1.2))
        self.pushButton_7.clicked.connect(lambda: self.scrollResize(0.8))

        # ??????????????????*********************************************
        self.models_dict = {'????????????????????????????????????(RGB)': 'weights/RGBbest.pt',
                            '?????????????????????????????????????????????(RGB)': 'weights/RGBbest.pt',
                            '?????????????????????????????????????????????(PAN)': 'weights/PANbest.pt',
                            '?????????????????????????????????????????????(IR)': 'weights/IRbest.pt'}

        # ??????????????????????????????
        self.comboBox.currentIndexChanged.connect(
            lambda: self.changemodel(self.models_dict[self.comboBox.currentText()])
        )

        # ?????????????????????
        self.rows = 2   # rows
        self.colums = 2   # columns

        # ???????????? ????????????????????????????????????????????????
        self.th = OutputThread(data=self)
        # ??????????????????GUI??????
        self.th.signalForText.connect(self.outputWritten)
        self.th.judge.connect(self.judge_thread)
        # ??????????????????????????????GUI?????????
        # sys.stdout = self.th
        # sys.stderr = self.th
        self.th.start()

        # ???????????????????????????
        self.train_win = TrainWindow()

        # ?????????????????????
        self.reset(start=True)

        # ????????????
        # self.qmain = queue.Queue()
        self.qmain = Queue()

        # ???????????????????????????????????????
        self.train_output_q = Queue()

        # ?????????????????????????????????
        self.time_queue = Queue()

        # ??????????????????
        self.current_task = None

    def judge_thread(self):
        '''
        ???????????????????????????????????????
        ???????????????????????????????????????????????????
        ??????????????????????????????????????????????????????????????????
        ?????????????????????????????????????????????????????????
        '''
        # ???????????????????????????????????????GUI?????????
        '''
        ???????????????????????????????????????
        1. ????????????dict????????????????????????????????????????????????batchs???epoch??????
        2. ????????????????????????????????????2??? ??????????????????????????????????????????????????????
        3. ???????????????????????????????????????????????????
        '''
        if not self.train_output_q.empty():
            s = self.train_output_q.get()
            # ????????????????????????????????????????????????????????????epoch
            if isinstance(s, dict):
                batchs = s['batchs']
                i = s['i']
                epoch = s['epoch']
                self.train_win.progressBar.setMaximum(batchs)
                self.train_win.progressBar.setValue(i)
                self.progressBar.setValue(epoch)
            # ????????????????????????????????????????????????????????????????????????????????????????????????
            # ??????????????????????????????????????????????????????????????????????????????????????????
            # ??????????????????
            elif len(s) == 2:
                # ?????????????????????????????????
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

                # ???????????????????????????????????????????????????
                s = s[0]
                self.train_win.textBrowser.append(s)
            else:
                self.train_win.textBrowser.append(s)
        # ?????????????????????
        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # ????????????????????????3?????????????????????????????????????????????????????????????????????????????????
        if self.time_queue.qsize() == 3:
            print('??????')
            string = self.time_queue.get()
            start = self.time_queue.get()
            end = self.time_queue.get()
            self.textBrowser.append(string + ('%.4f') % (end-start) + 's')
        # ??????????????????
        # ??????????????????
        if self.qmain.empty():
            return
        # ????????????????????????
        else:
            self.current_task = self.qmain.get()
            self.current_task.start()

    def changemodel(self, model_weight):
        '''
        ???????????????????????????????????????
        :param: model_weight: ???????????????????????????, str, ??????
        '''
        self.model_weight = model_weight
        print(self.model_weight)

    def choosemodel(self):
        '''
        ?????????????????????????????????
        '''
        selected_filter = "Models (*.pt);;All Files(*)"
        FileName, FileType = QFileDialog.getOpenFileName(self, "????????????", "", selected_filter)
        # ????????????????????????????????????????????????????????????????????????????????????
        if FileName != '':
            self.model_weight = FileName.replace(os.getcwd()+'/', '', 1)
            # print(self.model_weight)

    def reset(self, start=False):
        '''
        ????????????

        ????????????????????????, ?????????

        :param: start: ???????????????????????????????????????????????????, bool
                    True: ???????????????????????????????????????
                    False?????????????????????????????????????????????????????????
        '''
        # ???????????????
        self.treeWidget.expandToDepth(0)

        # ??????????????????
        self.tableWidget.setRowCount(0)

        # ?????????????????????
        self.images = []
        self.sourceimage = ''
        self.result = []

        # ?????????????????????
        self.textBrowser.clear()
        # self.textBrowser_2.clear()

        # ????????????
        init_dir(self.config['output_dir'])
        init_dir(self.config['source'])
        init_dir(self.config['temp'])
        init_dir(self.config['result'])

        # ???scrollArea????????????????????????
        self.last_time_ymove = 0
        self.last_time_xmove = 0
        self.scrollArea.installEventFilter(self)

        # ???????????????????????????
        self.spinBox.setValue(self.colums)
        self.spinBox_2.setValue(self.rows)
        # ??????????????????????????????
        self.spinBox.valueChanged.connect(self.changevalue)
        self.spinBox_2.valueChanged.connect(self.changevalue)

        # ???????????????????????????????????????
        if start:
            # ?????????????????????
            self.doubleSpinBox.setValue(0.0)
            self.doubleSpinBox_2.setValue(0.0)

            # ??????????????????
            self.comboBox.setCurrentText('????????????????????????????????????(RGB)')
            self.model_weight = self.models_dict['????????????????????????????????????(RGB)']
        else:
            # ????????????????????????
            showImages(self.gridLayout_2, self.colums, self.rows, [])

    def default(self):
        '''
        ??????????????????????????????
        '''
        self.reset(start=True)

    def exitprogram(self):
        '''
        ????????????
        '''
        try:
            # ?????????????????????????????????
            self.train_pro.kill()
            self.train_pro.close()
        except Exception as e:
            print(e)
        exit()

    def outputWritten(self, text):
        '''
        ?????????????????????
        '''
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def scrollResize(self, rate):
        '''
        ????????????????????????
        '''
        width = self.widget_10.width() * rate
        height = self.widget_10.height() * rate
        self.widget_10.resize(int(width), int(height))

    def changevalue(self):
        '''
        ????????????????????????????????????????????????
        '''
        self.colums = self.spinBox.value()
        self.rows = self.spinBox_2.value()

    def onclick(self, option):
        '''
        ?????????????????????????????????????????????
        '''
        # ????????????????????????????????????????????????????????????
        if self.current_task is not None:
            QMessageBox.warning(self, "??????", "?????????????????????????????????")
            return
        try:
            # ????????????????????????????????????????????????????????????
            getattr(self, self.options[option])()
        except Exception as e:
            print(e)

    def openfile(self):
        '''
        ??????????????????
        '''
        self.tableWidget.setRowCount(0)
        # ??????????????????????????????????????????????????????????????????????????????
        selected_filter = "Images (*.png *.jpg *.JPEG);;All Files(*)"
        imgName, imgType = QFileDialog.getOpenFileName(self, "????????????", "", selected_filter)
        # ??????????????????????????????????????????
        if len(imgName) == 0:
            return
        else:
            '''
            ???sourceimage???images?????????????????????????????????,
            ??????showImages????????????
            '''
            self.reset(start=False)
            # ???????????????????????????
            imgName = imgName.replace(os.getcwd()+'/', '', 1)
            # ??????source??????????????????????????????source???
            init_dir(self.config['source'])
            init_dir(self.config['temp'])
            init_dir(self.config['result'])
            self.sourceimage = shutil.copy(imgName, self.config['source'])
            image = shutil.copy(self.sourceimage, self.config['temp'])

            self.images = [image]
            showImages(self.gridLayout_2, self.colums, self.rows, self.images)

    def detectdir(self):
        '''
        ????????????????????????????????????
        '''
        select = QFileDialog.getExistingDirectory(self, "???????????????", "")
        # ?????????????????????????????????
        if len(select) == 0 or os.path.exists(select) is False:
            return
        directory = select.replace(os.getcwd()+'/', '', 1)
        # ?????????????????????????????????????????????????????????
        outputdir = self.config['batchoutput']
        # ???????????????
        self.detectdir_th = DetectThread([directory, self.model_weight, outputdir])

        self.detectdir_th.finished.connect(self.detectdir_result)
        self.qmain.put(self.detectdir_th)
        self.time_queue.put('??????????????????: ')
        self.time_queue.put(time.time())

    def detectdir_result(self):
        '''
        ?????????????????????????????????
        '''
        self.time_queue.put(time.time())
        self.textBrowser.append('???????????????????????????: ' + self.config['batchoutput'])
        self.current_task = None

    def cropimage(self):
        '''
        ????????????
        ?????????????????????self.images??????????????????
        ????????????????????????????????????self.images??????????????????????????????
        '''
        # ????????????????????????????????????????????????
        if len(self.images) != 1 and len(self.result) > 0:
            QMessageBox.warning(self, "??????", "??????????????????????????????")
            return
        else:
            self.images = [self.sourceimage]
        # ????????????????????????
        overlap_h = self.doubleSpinBox.value()
        overlap_v = self.doubleSpinBox_2.value()
        # ???????????????????????????
        result = cropimage_overlap(self.images[0], self.colums,
                                   self.rows, self.config['temp'],
                                   h=overlap_h, v=overlap_v)
        # ???????????????????????????????????????????????????????????????????????????????????????
        self.images = result[0]
        show_name = result[1]
        # ????????????
        # showImages(self.gridLayout_2, self.colums, self.rows, self.images)
        showImages(self.gridLayout_2, self.colums, self.rows, [show_name])
        self.tableWidget.setRowCount(0)

    def enlighten(self):
        '''
        self.images(?????????)  ->  self.images???????????????
        showImages ??????
        '''
        if len(self.images) == 0:
            QMessageBox.warning(self, "??????", "??????????????????")
            return
        # ????????????????????????????????????
        # ?????????????????????QObject????????????moveToThread???????????????
        self.lighten_th = QThread()
        self.lighten_wk = EnlightenWork(self.images)
        self.lighten_wk.moveToThread(self.lighten_th)

        # ????????????????????????
        self.lighten_th.started.connect(self.lighten_wk.run)
        self.lighten_wk.finished.connect(self.lighten_th.quit)
        self.lighten_wk.finished.connect(self.lighten_wk.deleteLater)
        self.lighten_th.finished.connect(self.lighten_th.deleteLater)

        self.lighten_wk.finished.connect(self.enlighten_finished)

        self.qmain.put(self.lighten_th)
        # ?????????????????????
        self.time_queue.put('??????????????????: ')
        self.time_queue.put(time.time())

    def enlighten_finished(self):
        '''
        ??????????????????????????????????????????????????????
        ????????????????????????
        '''
        lowlight_result = self.config['temp']
        image_names = [(lowlight_result + '/' + name) for name in os.listdir(lowlight_result)]
        image_names.sort()
        self.images = image_names
        print('enlighten result:', self.images)
        showImages(self.gridLayout_2, self.colums, self.rows, self.images)
        self.tableWidget.setRowCount(0)
        self.current_task = None
        self.time_queue.put(time.time())

    def ridfog(self):
        '''
        ?????????????????????__init__.py???
        self.images -> self.images
        ?????????  ->  ?????????
        '''
        paths = self.config['fog_dir']
        target = paths[0]
        out_result = self.config['fog_outdoor']
        init_dir(target)
        init_dir(out_result)
        if len(self.images) == 0:
            return
        # ???????????????????????????????????????????????????
        for file in self.images:
            try:
                shutil.copy(file, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except Exception:
                print("Unexpected error:", sys.exc_info())
        # ?????????
        self.ridfog_th = FogThread(paths)
        self.ridfog_th.finished.connect(self.ridfog_result)
        self.qmain.put(self.ridfog_th)
        self.time_queue.put('????????????: ')
        self.time_queue.put(time.time())

    def ridfog_result(self):
        '''
        ???????????????????????????????????????????????????
        ??????????????????????????????
        '''
        out_result = self.config['fog_outdoor']
        image_names = [(out_result + '/' + name) for name in os.listdir(out_result)]
        image_names.sort()
        init_dir(self.config['temp'])
        results = []
        # ???????????????????????????????????????????????????????????????
        for image in image_names:
            results.append(shutil.copy(image, self.config['temp']))
        self.images = results
        print('remove fog result:', self.images)
        showImages(self.gridLayout_2, self.colums, self.rows, self.images)
        self.tableWidget.setRowCount(0)
        self.time_queue.put(time.time())
        self.current_task = None

    def concat(self):
        '''
        ??????????????????

        ??????????????????????????????
        1. ???????????????????????????????????????
        2. ?????????????????????????????????????????????
        ????????????????????????????????????????????????????????????????????????????????????????????????
        ???????????????????????????????????????????????????
        '''
        # ???????????????
        n = len(self.result)
        # ???????????????
        m = len(self.images)
        # ???????????????????????????????????????????????????
        if m > 1 and n > 1:
            concat_path = self.config['result']
            # ????????????????????????????????????????????????????????????????????????
            # concat_path = self.config['temp']
        # ?????????????????????????????????
        elif m > 1:
            concat_path = self.config['temp']
        # ?????????????????????
        else:
            return
        image_names = []
        for name in os.listdir(concat_path):
            # ??????txt??????
            if name[-4:] != '.txt':
                image_names.append(concat_path + '/' + name)
        image_names.sort()
        # ???????????????????????????
        img_result = concat_image(image_names, self.colums, self.rows,
                                  overlap_h=self.doubleSpinBox.value(),
                                  overlap_v=self.doubleSpinBox_2.value())
        # ??????????????????????????????????????????????????????
        init_dir(concat_path)
        cv2.imwrite(image_names[0], img_result)
        # ???????????????????????????self.images?????????????????????
        # ?????????????????????????????????self.result????????????????????????
        if concat_path == self.config['temp']:
            self.images = [image_names[0]]
        else:
            self.result = [image_names[0]]

        showImages(self.gridLayout_2, self.colums, self.rows, [image_names[0]])

    def detection(self):
        '''
        ???????????????????????????????????????
        ???????????????????????????
        '''
        if len(self.images) == 0:
            QMessageBox.warning(self, "??????", "??????????????????????????????")
            return
        detect_dir = os.path.dirname(self.images[0])
        self.detect_th = DetectThread([detect_dir, self.model_weight])

        self.detect_th.finished.connect(self.detect_result)
        self.qmain.put(self.detect_th)
        self.time_queue.put('????????????: ')
        self.time_queue.put(time.time())

    def detect_result(self, result):
        '''
        ???????????????????????????
        '''
        self.result = result
        init_dir(self.config['result'])
        result0 = []
        # ????????????????????????????????????????????????,?????????
        for image in self.result:
            result0.append(shutil.copy(image, self.config['result']))
        self.result = result0
        showImages(self.gridLayout_2, self.colums, self.rows, self.result)
        print_txt(self.tableWidget, self.config['targets'], result)
        self.time_queue.put(time.time())
        self.detect_th.deleteLater()
        self.current_task = None

    def selectdata(self):
        '''
        ??????????????????
        '''
        self.train_win.show()
        self.train_win.exec_()

    def changetrain(self):
        '''
        ???????????????????????????????????????,??????????????????
        '''
        if self.pushButton_4.text() == '????????????':
            self.train()
        else:
            self.stoptrain()

    def showtrain(self):
        '''
        ???????????????????????????
        '''
        self.train_win.show()
        self.train_win.exec()

    def stoptrain(self):
        '''
        ??????????????????
        '''
        try:
            self.train_pro.kill()
            self.train_pro.close()
        except Exception as e:
            print(e)
        self.pushButton_4.setText('????????????')
        print('stoptrain')

    def train(self):
        '''
        ????????????
        '''
        # ??????????????????????????????
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
        self.pushButton_4.setText('????????????')
        self.progressBar.setMaximum(epochs - 1)
        self.train_win.label_3.setText('epoch:0/' + str(epochs-1))

    def lightweight(self):
        '''
        ???????????????????????????????????????
        '''
        self.model_weight = 'weights/Fast.pt'

    def eventFilter(self, obj, event):
        '''
        ?????????????????????????????????
        '''
        if event.type() == QEvent.MouseMove:
            if self.last_time_ymove == 0:
                self.last_time_ymove = event.pos().y()
            if self.last_time_xmove == 0:
                self.last_time_xmove = event.pos().x()
            distance_y = self.last_time_ymove - event.pos().y()
            distance_x = self.last_time_xmove - event.pos().x()
            self.scrollArea.verticalScrollBar().setValue(
                self.scrollArea.verticalScrollBar().value() + distance_y
            )
            self.scrollArea.horizontalScrollBar().setValue(
                self.scrollArea.horizontalScrollBar().value() + distance_x
            )
            self.last_time_ymove = event.pos().y()
            self.last_time_xmove = event.pos().x()
        elif event.type() == QEvent.MouseButtonRelease:
            self.last_time_ymove = 0
            self.last_time_xmove = 0
        # return QWidget.eventFilter(self, source, event)
        return super(MyWindow, self).eventFilter(obj, event)

    def closeEvent(self, event):
        sys.exit(app.exec_())


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    myWin = MyWindow()
    # ??????qdarkstyle??????
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    myWin.show()
    sys.exit(app.exec_())
