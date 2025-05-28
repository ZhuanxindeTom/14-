from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtCore import *
from win.yolov5 import Ui_MainWindow
import sys, os, time, datetime
from detect_plate import *
from PyQt5.QtGui import QImage, QPixmap
import csv
import argparse
import numpy as np
import torch

class myMainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(myMainWindow, self).__init__()
        self.setupUi(self)
        self.timer_camera = QTimer()
        self.flag = True
        self.flag_1 = False
        self.file_type = None
        self.plate_data = []  # 存储车牌数据

        self.PB1.clicked.connect(self.OpenFile)  # 打开文件
        self.PB2.setText("导出csv")
        self.PB2.clicked.connect(self.ExportToCSV)  # 导出CSV
        self.PB3.clicked.connect(self.RunOrContinue)  # 开始/暂停检测
        self.PB4.clicked.connect(self.Stop)  # 结束检测

        self.det_thread = DetThread()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_1))
        self.det_thread.send_result.connect(lambda x: self.statistic_msg(x))
        self.det_thread.send_plate_data.connect(self.collect_plate_data)

    def collect_plate_data(self, plate_result):
        """收集车牌数据并添加时间戳"""
        if plate_result:  # 过滤空结果
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.plate_data.append([timestamp, plate_result])

    def ExportToCSV(self):
        """导出CSV文件"""
        if not self.plate_data:
            QMessageBox.warning(self, "警告", "没有检测到车牌数据！")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "", "CSV Files (*.csv)")

        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['检测时间', '车牌号码'])
                    writer.writerows(self.plate_data)
                QMessageBox.information(self, "成功", f"数据已保存至：{path}")
                self.plate_data = []  # 导出后清空数据
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def OpenFile(self):
        self.name, _ = QFileDialog.getOpenFileName(
            self, "打开文件", "./imgs/", "All Files(*);;*.jpg;;*.png;;*.mp4;;*.avi")
        _, self.file_type = os.path.splitext(self.name)
        if self.name:
            if self.file_type.lower() in ['.jpg', '.png']:
                ori_img = cv2.imread(self.name)
                self.show_image(ori_img, self.label_1)
            elif self.file_type.lower() in ['.mp4', '.avi']:
                cap = cv2.VideoCapture(self.name)
                ret, frame = cap.read()
                if ret:
                    self.show_image(frame, self.label_1)
            self.det_thread.path = self.name

    def RunOrContinue(self):
        self.det_thread.jump_out = False
        if not self.det_thread.isRunning():
            if self.file_type and self.file_type.lower() in ['.jpg', '.png', '.mp4', '.avi']:
                self.det_thread.start()
                self.statistic_msg('开始检测>>>')
            else:
                QMessageBox.warning(self, "提示", "请先选择有效文件！")

    def Stop(self):
        self.det_thread.jump_out = True
        self.flag_1 = False
        self.file_type = None
        self.statistic_msg('结束检测>>>')

    def statistic_msg(self, msg):
        self.textBrowser.append(msg)

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            scal = min(w / iw, h / ih)
            nw, nh = int(iw * scal), int(ih * scal)
            img_resized = cv2.resize(img_src, (nw, nh))
            frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0],
                         frame.strides[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(f"图像显示错误: {str(e)}")


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_result = pyqtSignal(str)
    send_plate_data = pyqtSignal(str)  # 新增车牌数据信号

    def __init__(self):
        super().__init__()
        self.path = None
        self.jump_out = False
        self.plate_result = None

        # 模型初始化
        self.detect_model = load_model(opt.detect_model, device)
        self.plate_rec_model = init_model(device, opt.rec_model, is_color=True)

    def run(self):
        total = sum(p.numel() for p in self.detect_model.parameters())
        total_1 = sum(p.numel() for p in self.plate_rec_model.parameters())
        self.send_result.emit(f"模型参数 - 检测: {total / 1e6:.2f}M, 识别: {total_1 / 1e6:.2f}M")

        if self.path:
            file_type = os.path.splitext(self.path)[1].lower()
            if file_type in ['.jpg', '.png']:
                self.process_image()
            elif file_type in ['.mp4', '.avi']:
                self.process_video()
        else:
            self.send_result.emit("错误：未选择有效文件路径")

    def process_image(self):
        try:
            img = cv2.imread(self.path)
            if img is None:
                raise ValueError("无法读取图像文件")
            result_list = detect_Recognition_plate(
                self.detect_model, img, device,
                self.plate_rec_model, opt.img_size, opt.is_color
            )
            ori_img, result = draw_result(img, result_list)
            self.send_img.emit(ori_img)
            if result:
                self.send_result.emit(f"检测结果：{result}")
                self.send_plate_data.emit(result)
            self.save_output(ori_img)
        except Exception as e:
            self.send_result.emit(f"图像处理错误：{str(e)}")

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output_path = os.path.join(opt.output, os.path.basename(self.path))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened() and not self.jump_out:
                ret, frame = cap.read()
                if not ret:
                    break
                result_list = detect_Recognition_plate(
                    self.detect_model, frame, device,
                    self.plate_rec_model, opt.img_size, opt.is_color
                )
                processed_frame, result = draw_result(frame, result_list)
                self.send_img.emit(processed_frame)
                writer.write(processed_frame)
                if result and result != self.plate_result:
                    self.send_plate_data.emit(result)
                    self.plate_result = result
            cap.release()
            writer.release()
        except Exception as e:
            self.send_result.emit(f"视频处理错误：{str(e)}")

    def save_output(self, img):
        output_dir = opt.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(self.path))
        cv2.imwrite(output_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='weights/plate_detect.pt')
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth')
    parser.add_argument('--is_color', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--output', type=str, default='result')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    app = QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())
