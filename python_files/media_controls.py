import os
import cv2
import time
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox, QPushButton, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from datetime import datetime
from detection import DetectionThread
from utils import Utils
from PyQt5.QtMultimedia import QMediaPlayer


class MediaControls:
    def __init__(self, ui_mainwindow):
        self.ui = ui_mainwindow
        self.original_geometry = self.ui.media_label.geometry()  # 保存初始几何形状
        self.auto_detect_enabled = False
        self.auto_detect_mode = True

    def open_media(self):
        if self.ui.video_capture is not None:
            self.ui.video_capture.release()
        if self.ui.media_player.state() == QMediaPlayer.PlayingState:
            self.ui.media_player.stop()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "打开媒体文件", "",
                                                   "媒体文件 (*.mp4 *.avi *.jpg *.png);;所有文件 (*)", options=options)
        if file_path:
            if self.ui.video_capture is not None:
                self.ui.frame_timer.stop()
                self.ui.video_capture.release()
                self.ui.video_capture = None

            if file_path.endswith(('.jpg', '.png')):
                self.ui.current_media_type = 'image'
                self.ui.current_image_path = file_path
                self.load_image(file_path)
            else:
                self.ui.current_media_type = 'video'
                self.ui.current_image_path = None
                self.play_video(file_path)

    def load_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.ui.media_label.setPixmap(pixmap)
        self.ui.media_player.stop()
        self.ui.media_label.show()
        self.ui.video_widget.hide()
        self.ui.is_playing = False
        image = cv2.imread(file_path)
        if image is not None:
            self.ui.current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ui.frame_timer.stop()

    def play_video(self, file_path):
        self.ui.statusbar.showMessage("视频加载中...")
        QApplication.processEvents()
        try:
            if self.ui.video_capture is not None:
                self.ui.video_capture.release()
                self.ui.statusbar.showMessage("视频加载完成")

            self.ui.video_capture = cv2.VideoCapture(file_path)
            if not self.ui.video_capture.isOpened():
                self.ui.show_error_message("错误：无法打开视频文件。")
                return

            self.ui.video_fps = int(self.ui.video_capture.get(cv2.CAP_PROP_FPS))
            self.ui.total_frames = int(self.ui.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.ui.current_frame_number = 0
            self.ui.progress_bar.setMaximum(self.ui.total_frames)
            self.ui.progress_bar.setValue(0)
            self.ui.progress_bar.show()
            self.ui.video_fps_spinbox.setValue(min(30, self.ui.video_fps))
            self.update_detection_fps(self.ui.detect_fps_spinbox.value())
            self.ui.frame_timer.start(int(1000 / self.ui.video_fps))
            self.ui.media_label.show()
            self.ui.video_widget.hide()
            self.ui.is_playing = True
            self.ui.detection_cache.clear()
            self.ui.frame_count = 0
            self.ui.last_detection_time = 0
            self.ui.statusbar.showMessage("视频加载完成")

        except Exception as e:
            self.ui.show_error_message(f"视频播放错误: {str(e)}")
            if self.ui.video_capture is not None:
                self.ui.video_capture.release()
                self.ui.video_capture = None

    def update_fps(self, value):
        self.ui.fps = value
        self.ui.fps_label.setText(f"帧率: {value} FPS")
        if self.ui.frame_timer.isActive():
            self.ui.frame_timer.setInterval(int(1000 / value))

    def play_pause_video(self):
        if self.ui.current_media_type != 'video' or self.ui.video_capture is None:
            return
        if self.ui.is_playing:
            self.ui.frame_timer.stop()
            self.ui.is_playing = False
        else:
            if self.ui.video_capture.isOpened():
                self.ui.frame_timer.start(int(1000 / self.ui.fps))
                self.ui.is_playing = True

    def detect_image(self):
        if self.ui.current_media_type == 'image':
            if self.ui.current_image_path:
                image = cv2.imread(self.ui.current_image_path)
                if image is not None:
                    self.ui.current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.detect_current_frame()
                else:
                    print("无法加载图片")
        elif self.ui.current_frame is not None:
            self.detect_current_frame()
        else:
            print("没有加载媒体文件")

    def detect_current_frame(self):
        if self.ui.current_frame is None or self.ui.is_detecting:
            return

        self.ui.is_detecting = True
        self.ui.progress_bar.show()

        self.detection_thread = DetectionThread(self.ui.current_model, self.ui.current_frame)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()

        try:
            if (self.ui.last_processed_frame is not None and
                    Utils.is_frame_similar(self.ui.current_frame, self.ui.last_processed_frame)):
                self.ui.is_detecting = False
                return

            self.ui.tableWidget.setRowCount(0)
            results = self.ui.current_model(self.ui.current_frame)
            detected_frame = self.ui.current_frame.copy()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    label = result.names[class_id]

                    color = Utils.get_color_for_label(label)
                    cv2.rectangle(detected_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    text = f"{label} {confidence:.2f}"
                    cv2.putText(detected_frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    self.update_table(label, confidence, x1, y1, x2, y2)

            self.ui.current_frame = detected_frame
            self.ui.last_processed_frame = self.ui.current_frame.copy()

        except Exception as e:
            self.ui.show_error_message(f"检测错误: {str(e)}")
        finally:
            self.ui.is_detecting = False

    def display_next_frame(self):
        if self.ui.video_capture is None or not self.ui.video_capture.isOpened():
            self.ui.frame_timer.stop()
            self.ui.is_playing = False
            return
        try:
            ret, frame = self.ui.video_capture.read()
            if not ret:
                self.ui.frame_timer.stop()
                self.ui.is_playing = False
                self.ui.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.ui.current_frame_number = 0
                self.ui.progress_bar.setValue(0)
                return
            self.ui.current_frame_number += 1
            self.ui.progress_bar.setValue(self.ui.current_frame_number)
            current_time = time.time()
            should_detect = (
                        self.auto_detect_enabled and current_time - self.ui.last_detection_time >= self.ui.detection_interval)
            if should_detect:
                self.ui.current_frame = frame
                self.detect_current_frame()
                self.ui.last_detection_time = current_time
                display_frame = self.ui.current_frame
            else:
                display_frame = frame
            self.display_frame(display_frame)
        except Exception as e:
            self.ui.show_error_message(f"帧处理错误: {str(e)}")
            self.ui.frame_timer.stop()
            self.ui.is_playing = False

    def display_frame(self, frame):
        if frame is None:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.media_label.setPixmap(pixmap)

    def on_detection_finished(self, results):
        self.ui.is_detecting = False
        detected_frame = self.ui.current_frame.copy()
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf)
                class_id = int(box.cls)
                label = result.names[class_id]

                color = Utils.get_color_for_label(label)
                cv2.rectangle(detected_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(detected_frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                self.update_table(label, confidence, x1, y1, x2, y2)
        self.ui.current_frame = detected_frame
        self.display_frame(detected_frame)

    def update_table(self, label, confidence, x1, y1, x2, y2):
        row_position = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(row_position)
        items = [
            QTableWidgetItem("检测"),
            QTableWidgetItem(label),
            QTableWidgetItem(f"{confidence:.2f}"),
            QTableWidgetItem(f"{int(x1)}"),
            QTableWidgetItem(f"{int(y1)}"),
            QTableWidgetItem(f"{int(x2 - x1)}"),
            QTableWidgetItem(f"{int(y2 - y1)}")
        ]
        for col, item in enumerate(items):
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.tableWidget.setItem(row_position, col, item)
        self.ui.insert_detection(label, confidence, int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def take_screenshot(self):
        if self.ui.current_frame is None:
            QMessageBox.warning(None, "警告", "没有可用的图像帧")
            return
        try:
            if not os.path.exists(self.ui.results_dir):
                os.makedirs(self.ui.results_dir)

            current_detections = []
            for row in range(self.ui.tableWidget.rowCount()):
                detection = {
                    'label': self.ui.tableWidget.item(row, 1).text(),
                    'confidence': float(self.ui.tableWidget.item(row, 2).text()),
                    'x': int(self.ui.tableWidget.item(row, 3).text()),
                    'y': int(self.ui.tableWidget.item(row, 4).text()),
                    'width': int(self.ui.tableWidget.item(row, 5).text()),
                    'height': int(self.ui.tableWidget.item(row, 6).text())
                }
                current_detections.append(detection)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{self.ui.results_dir}/screenshot_{timestamp}.png"
            info_filename = f"{self.ui.results_dir}/screenshot_{timestamp}_info.txt"

            cv2.imwrite(image_filename, cv2.cvtColor(self.ui.current_frame, cv2.COLOR_RGB2BGR))

            with open(info_filename, 'w', encoding='utf-8') as f:
                f.write(f"截图时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"当前帧: {self.ui.current_frame_number}\n\n")
                f.write("检测结果:\n")
                for det in current_detections:
                    f.write(f"标签: {det['label']}\n")
                    f.write(f"置信度: {det['confidence']:.2f}\n")
                    f.write(f"位置: X={det['x']}, Y={det['y']}\n")
                    f.write(f"大小: {det['width']}x{det['height']}\n")
                    f.write("-" * 30 + "\n")

            QMessageBox.information(None, "成功",
                                    f"截图已保存至:\n{image_filename}\n检测信息已保存至:\n{info_filename}")

        except Exception as e:
            QMessageBox.critical(None, "错误", f"保存截图时发生错误: {str(e)}")

    def export_results(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(self.ui.results_dir, f"export_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            csv_filename = os.path.join(export_dir, "detection_results.csv")
            summary_filename = os.path.join(export_dir, "summary.txt")

            cursor = self.ui.conn.cursor()
            cursor.execute('''SELECT label, confidence, x, y, width, height, timestamp
                              FROM detections
                              ORDER BY timestamp DESC''')
            results = cursor.fetchall()
            total_detections = len(results)
            labels_count = {}
            avg_confidence = 0

            with open(csv_filename, 'w', encoding='utf-8') as f:
                f.write("标签,置信度,X,Y,宽度,高度,时间戳\n")
                for row in results:
                    label = row[0]
                    confidence = row[1]
                    labels_count[label] = labels_count.get(label, 0) + 1
                    avg_confidence += confidence
                    f.write(f"{label},{confidence:.3f},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")

            avg_confidence = avg_confidence / total_detections if total_detections > 0 else 0

            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(f"检测结果统计报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\n总检测数量: {total_detections}\n")
                f.write(f"平均置信度: {avg_confidence:.3f}\n\n")
                f.write("各类别检测数量:\n")
                for label, count in labels_count.items():
                    f.write(f"{label}: {count}\n")

            QMessageBox.information(None, "成功",
                                    f"检测结果已导出至目录:\n{export_dir}\n\n包含:\n- 详细检测数据 (CSV)\n- 统计摘要 (TXT)")

        except Exception as e:
            self.ui.show_error_message(f"导出错误: {str(e)}")

    def update_video_fps(self, value):
        self.ui.video_fps = value
        if self.ui.frame_timer.isActive():
            self.ui.frame_timer.setInterval(int(1000 / value))

    def progress_bar_click(self, event):
        if self.ui.video_capture is None:
            return
        self.ui.is_dragging = True
        width = self.ui.progress_bar.width()
        x = event.pos().x()
        target_frame = int((x / width) * self.ui.total_frames)
        self.seek_to_frame(target_frame)

    def progress_bar_drag(self, event):
        if self.ui.is_dragging and self.ui.video_capture is not None:
            width = self.ui.progress_bar.width()
            x = max(0, min(event.pos().x(), width))
            target_frame = int((x / width) * self.ui.total_frames)
            self.seek_to_frame(target_frame)

    def progress_bar_release(self, event):
        self.ui.is_dragging = False

    def seek_to_frame(self, frame_number):
        if self.ui.video_capture is None:
            return
        frame_number = max(0, min(frame_number, self.ui.total_frames - 1))
        self.ui.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.ui.current_frame_number = frame_number
        self.ui.progress_bar.setValue(frame_number)
        ret, frame = self.ui.video_capture.read()
        if ret:
            self.ui.current_frame = frame
            self.display_frame(frame)
            self.ui.last_frame_position = frame_number

    def update_detection_fps(self, value):
        self.ui.detection_fps = value
        self.ui.detection_interval = 1.0 / value
        self.ui.frames_per_detection = max(1, int(self.ui.video_fps / value))
        print(f"更新检测间隔: 每{self.ui.frames_per_detection}帧检测一次")

    def update_detection_interval(self, value):
        self.ui.detection_interval = value

    def change_font(self, font):
        for button in self.ui.centralwidget.findChildren(QPushButton):
            button.setFont(font)

    def adjust_vertical(self, value):
        current_geometry = self.ui.media_label.geometry()
        new_y = 20 + (340 - self.ui.media_label.height()) * value / 100
        self.ui.media_label.setGeometry(current_geometry.x(), int(new_y), current_geometry.width(),
                                        current_geometry.height())

    def adjust_horizontal(self, value):
        current_geometry = self.ui.media_label.geometry()
        new_x = 170 + (440 - self.ui.media_label.width()) * value / 100
        self.ui.media_label.setGeometry(int(new_x), current_geometry.y(), current_geometry.width(),
                                        current_geometry.height())

    def change_model(self, index):
        if index == 0:
            self.ui.current_model = self.ui.yolo_model
        elif index == 1:
            self.ui.current_model = self.ui.duo_model
        elif index == 2:
            self.ui.current_model = self.ui.duo2_model
        elif index == 3:
            self.ui.current_model = self.ui.duo3_model
        elif index == 4:
            self.ui.current_model = self.ui.duo4_model
        elif index == 5:
            self.ui.current_model = self.ui.duo5_model
        print(f"切换到模型: {self.ui.mod_change_button.currentText()}")

    def close_event(self, event):
        self.ui.frame_timer.stop()
        if self.ui.video_capture is not None:
            self.ui.video_capture.release()
        self.ui.conn.close()
        self.ui.detection_cache.clear()
        event.accept()

    def zoom_in(self):
        current_size = self.ui.media_label.size()
        new_size = current_size * 1.2
        self.ui.media_label.resize(new_size)

    def zoom_out(self):
        current_size = self.ui.media_label.size()
        new_size = current_size * 0.8
        self.ui.media_label.resize(new_size)

    def reset_view(self):
        self.ui.media_label.setGeometry(self.original_geometry)

    def toggle_auto_detect(self, enabled):
        self.auto_detect_enabled = enabled

    def toggle_auto_detect_mode(self):
        self.auto_detect_mode = not self.auto_detect_mode
        if self.auto_detect_mode:
            self.ui.toggle_auto_detect_button.setText('停止自动检测')