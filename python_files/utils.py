import cv2

class Utils:
    @staticmethod
    def get_color_for_label(label):
        label_colors = {
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'truck': (0, 0, 255),
        }
        if label not in label_colors:
            hash_value = hash(label)
            return (hash_value % 255, (hash_value * 2) % 255, (hash_value * 3) % 255)
        return label_colors[label]

    @staticmethod
    def is_frame_similar(frame1, frame2, threshold=0.95):
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
            return score > threshold
        except:
            return False