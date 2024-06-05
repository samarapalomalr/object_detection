import unittest
import cv2

class TestObjectDetection(unittest.TestCase):
    def test_video_capture(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to capture video frame.")

if __name__ == '__main__':
    unittest.main()
