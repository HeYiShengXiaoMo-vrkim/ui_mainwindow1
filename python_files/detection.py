from PyQt5.QtCore import QThread, pyqtSignal

class DetectionThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, model, frame):
        super().__init__()
        self.model = model
        self.frame = frame

    def run(self):
        try:
            results = self.model(self.frame)
            self.finished.emit(results)
        except Exception as e:
            print(f"检测错误: {str(e)}")