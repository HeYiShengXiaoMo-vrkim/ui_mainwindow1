import os
import cv2
import time
import numpy as np
import sqlite3
import threading
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel, QFileDialog, QPushButton, QSlider, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO
from media_controls import MediaControls
from detection import DetectionThread
from database import Database
from utils import Utils

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.fps = 30
        self.video_capture = None
        self.frame_timer = None
        self.results_dir = "detection_results"
        self.detection_cache = {}
        self.current_media_type = None
        self.current_image_path = None
        self.is_playing = False
        self.is_detecting = False
        self.current_frame_number = 0
        self.total_frames = 0
        self.detection_interval = 1.0
        self.last_detection_time = 0
        self.is_dragging = False
        self.last_frame_position = 0
        self.last_processed_frame = None
        self.detection_fps = 2
        self.frames_per_detection = 15
        self.auto_detect = False

        self.database = Database()
        self.utils = Utils()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("yolov8")
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(800, 700)  # 调整窗口高度以适应新增按钮
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.frame_timer = QtCore.QTimer()
        self.frame_timer.timeout.connect(self.display_next_frame)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.error.connect(self.handle_media_error)

        self.video_widget = QVideoWidget(self.centralwidget)
        self.video_widget.setGeometry(QtCore.QRect(170, 20, 440, 340))

        self.media_label = QLabel(self.centralwidget)
        self.media_label.setGeometry(QtCore.QRect(170, 20, 440, 340))
        self.media_label.setScaledContents(True)

        self.slider_vertical = QSlider(self.centralwidget)
        self.slider_vertical.setGeometry(QtCore.QRect(620, 20, 20, 340))
        self.slider_vertical.setOrientation(Qt.Vertical)

        self.slider_horizontal = QSlider(self.centralwidget)
        self.slider_horizontal.setGeometry(QtCore.QRect(170, 370, 440, 20))
        self.slider_horizontal.setOrientation(Qt.Horizontal)

        self.fontComboBox = QtWidgets.QFontComboBox(self.centralwidget)
        self.fontComboBox.setGeometry(QtCore.QRect(20, 20, 121, 22))

        self.add_video = QPushButton("添加媒体", self.centralwidget)
        self.add_video.setGeometry(QtCore.QRect(20, 50, 121, 71))

        self.video_bigger = QPushButton("+", self.centralwidget)
        self.video_bigger.setGeometry(QtCore.QRect(20, 130, 51, 51))

        self.video_smaller = QPushButton("-", self.centralwidget)
        self.video_smaller.setGeometry(QtCore.QRect(90, 130, 51, 51))

        self.reset_button = QPushButton("复位", self.centralwidget)  # 添加复位按钮
        self.reset_button.setGeometry(QtCore.QRect(20, 190, 121, 31))

        self.mod_change_button = QtWidgets.QComboBox(self.centralwidget)
        self.mod_change_button.setGeometry(QtCore.QRect(20, 230, 121, 31))
        self.mod_change_button.addItems(["yolov8n模型", "yolov8l模型", "yolov8m模型", "yolov8s模型", "yolov8x模型"])

        self.detect_button = QPushButton('检测', self.centralwidget)
        self.detect_button.setGeometry(QtCore.QRect(20, 270, 121, 31))

        self.play_pause_button = QPushButton('播放/暂停', self.centralwidget)
        self.play_pause_button.setGeometry(QtCore.QRect(20, 310, 121, 31))

        self.auto_detect_button = QPushButton('自动检测', self.centralwidget)
        self.auto_detect_button.setGeometry(QtCore.QRect(20, 350, 121, 31))

        self.fps_slider = QSlider(Qt.Horizontal, self.centralwidget)
        self.fps_slider.setGeometry(QtCore.QRect(20, 390, 121, 20))
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(30)
        self.fps_slider.valueChanged.connect(self.update_fps)

        self.fps_label = QLabel("帧率: 30 FPS", self.centralwidget)
        self.fps_label.setGeometry(QtCore.QRect(20, 410, 121, 20))

        self.create_controls()

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(170, 400, 621, 250))
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(['类型', '对象', '置信度', 'X', 'Y', '宽度', '高度'])

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.yolo_model = YOLO(r"E:\python代码\underwater_detection_files\underwaterCode\underwaterCode\pt\yolov8n.pt")
        self.duo_model = YOLO(r"E:\python代码\underwater_detection_files\underwaterCode\underwaterCode\pt\yolov8l.pt")
        self.duo2_model = YOLO(r"E:\python代码\underwater_detection_files\underwaterCode\underwaterCode\pt\yolov8m.pt")
        self.duo3_model = YOLO(r"E:\python代码\underwater_detection_files\underwaterCode\underwaterCode\pt\yolov8s.pt")
        self.duo4_model = YOLO(r"E:\python代码\underwater_detection_files\underwaterCode\underwaterCode\pt\yolov8x.pt")
        self.current_model = self.yolo_model

        self.detection_lock = threading.Lock()

        self.conn = sqlite3.connect('detections.db')
        self.create_table()

        self.screenshot_button = QPushButton('截图', self.centralwidget)
        self.screenshot_button.setGeometry(QtCore.QRect(530, 320, 100, 25))
        self.screenshot_button.clicked.connect(self.take_screenshot)

        self.progress_bar = QProgressBar(self.centralwidget)
        self.progress_bar.setGeometry(QtCore.QRect(170, 380, 440, 10))
        self.progress_bar.hide()

        self.export_button = QPushButton('导出检测结果', self.centralwidget)
        self.export_button.setGeometry(QtCore.QRect(640, 320, 100, 25))
        self.export_button.clicked.connect(self.export_results)

        self.interval_spinbox = QSpinBox(self.centralwidget)
        self.interval_spinbox.setGeometry(QtCore.QRect(20, 440, 121, 25))
        self.interval_spinbox.setMinimum(1)
        self.interval_spinbox.setMaximum(30)
        self.interval_spinbox.setValue(5)
        self.interval_spinbox.valueChanged.connect(self.update_detection_interval)

        self.video_fps_label = QLabel("视频帧率:", self.centralwidget)
        self.video_fps_label.setGeometry(QtCore.QRect(20, 470, 60, 25))

        self.video_fps_spinbox = QSpinBox(self.centralwidget)
        self.video_fps_spinbox.setGeometry(QtCore.QRect(85, 470, 50, 25))
        self.video_fps_spinbox.setMinimum(1)
        self.video_fps_spinbox.setMaximum(60)
        self.video_fps_spinbox.setValue(30)
        self.video_fps_spinbox.valueChanged.connect(self.update_video_fps)

        self.detect_fps_label = QLabel("检测帧率:", self.centralwidget)
        self.detect_fps_label.setGeometry(QtCore.QRect(20, 500, 60, 25))

        self.detect_fps_spinbox = QSpinBox(self.centralwidget)
        self.detect_fps_spinbox.setGeometry(QtCore.QRect(85, 500, 50, 25))
        self.detect_fps_spinbox.setMinimum(1)
        self.detect_fps_spinbox.setMaximum(10)
        self.detect_fps_spinbox.setValue(2)
        self.detect_fps_spinbox.valueChanged.connect(self.update_detection_fps)

        self.interval_label = QLabel("检测间隔(秒):", self.centralwidget)
        self.interval_label.setGeometry(QtCore.QRect(20, 530, 80, 25))

        self.interval_spinbox = QDoubleSpinBox(self.centralwidget)
        self.interval_spinbox.setGeometry(QtCore.QRect(100, 530, 60, 25))
        self.interval_spinbox.setMinimum(0.1)
        self.interval_spinbox.setMaximum(5.0)
        self.interval_spinbox.setSingleStep(0.1)
        self.interval_spinbox.setValue(1.0)
        self.interval_spinbox.valueChanged.connect(self.update_detection_interval)

        # 初始化 MediaControls
        self.media_controls = MediaControls(self)

        self.add_video.clicked.connect(self.open_media)
        self.video_bigger.clicked.connect(self.media_controls.zoom_in)
        self.video_smaller.clicked.connect(self.media_controls.zoom_out)
        self.reset_button.clicked.connect(self.reset_view)  # 连接复位按钮的槽函数
        self.auto_detect_button.clicked.connect(self.toggle_auto_detect)  # 连接自动检测按钮的槽函数
        self.detect_button.clicked.connect(self.detect_image)
        self.fontComboBox.currentFontChanged.connect(self.change_font)
        self.slider_vertical.valueChanged.connect(self.adjust_vertical)
        self.slider_horizontal.valueChanged.connect(self.adjust_horizontal)
        self.mod_change_button.currentIndexChanged.connect(self.change_model)
        self.play_pause_button.clicked.connect(self.play_pause_video)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "underwater_detection"))

    def create_controls(self):
        self.screenshot_button = QPushButton('截图', self.centralwidget)
        self.screenshot_button.setGeometry(QtCore.QRect(530, 320, 100, 25))
        self.screenshot_button.clicked.connect(self.take_screenshot)

        self.export_button = QPushButton('导出检测结果', self.centralwidget)
        self.export_button.setGeometry(QtCore.QRect(640, 320, 100, 25))
        self.export_button.clicked.connect(self.export_results)

    def open_media(self):
        self.media_controls.open_media()

    def load_image(self, file_path):
        self.media_controls.load_image(file_path)

    def play_video(self, file_path):
        self.media_controls.play_video(file_path)

    def update_fps(self, value):
        self.media_controls.update_fps(value)

    def play_pause_video(self):
        self.media_controls.play_pause_video()

    def detect_image(self):
        self.media_controls.detect_image()

    def display_next_frame(self):
        self.media_controls.display_next_frame()

    def display_frame(self, frame):
        self.media_controls.display_frame(frame)

    def handle_media_error(self, error):
        error_msg = self.media_player.errorString()
        self.show_error_message(f"媒体播放器错误: {error_msg}")

    def show_error_message(self, message):
        QtWidgets.QMessageBox.critical(None, "错误", message)

    def create_table(self):
        self.database.create_table()

    def insert_detection(self, label, confidence, x, y, width, height):
        self.database.insert_detection(label, confidence, x, y, width, height)

    def clear_database(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM detections")
        self.conn.commit()

    def closeEvent(self, event):
        self.clear_database()
        self.media_controls.close_event(event)
        self.database.close()

    def take_screenshot(self):
        self.media_controls.take_screenshot()

    def export_results(self):
        self.media_controls.export_results()

    def change_font(self, font):
        self.media_controls.change_font(font)

    def adjust_vertical(self, value):
        self.media_controls.adjust_vertical(value)

    def adjust_horizontal(self, value):
        self.media_controls.adjust_horizontal(value)

    def change_model(self, index):
        self.media_controls.change_model(index)

    def progress_bar_click(self, event):
        self.media_controls.progress_bar_click(event)

    def progress_bar_drag(self, event):
        self.media_controls.progress_bar_drag(event)

    def progress_bar_release(self, event):
        self.media_controls.progress_bar_release(event)

    def update_detection_fps(self, value):
        self.media_controls.update_detection_fps(value)

    def update_detection_interval(self, value):
        self.media_controls.update_detection_interval(value)

    def update_video_fps(self, value):
        self.media_controls.update_video_fps(value)

    def reset_view(self):
        self.media_controls.reset_view()  # 调用 media_controls 的复位方法

    def toggle_auto_detect(self):
        self.auto_detect = not self.auto_detect
        if self.auto_detect:
            self.auto_detect_button.setText('停止自动检测')
        else:
            self.auto_detect_button.setText('自动检测')
        self.media_controls.toggle_auto_detect(self.auto_detect)