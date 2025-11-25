import sys
from server.logger_ser import logger
import cv2
from common import nms, perspective_img
import onnxruntime as ort
# from common import non_max_suppression
import numpy as np
import io
from PIL import Image
import io

class Detect_Allign:
    def __init__(self):
        self.session = self.__load_onnx_session()

    def __load_onnx_session(
        self,
        onnx_path = 'onnx_checkpoints/last.onnx'
    ):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session
    
    def detect_corners(self, img):
        w_original, h_original = img.shape[0], img.shape[1]
        # input shape [1, 3, 640, 640]
        # input type tensor(float)

        img = cv2.resize(img, (640,640))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)/255.0
        img = np.expand_dims(img,axis=0)
        results = self.session.run(None, {self.session.get_inputs()[0].name: img})
        pred = nms(results[0][0])
        tl, tr, bl,br = 0,0,0,0
        for (x,y,w,h, conf, id) in pred:
            x = x/640 * h_original
            y = y/640 * w_original
            if id == 0 and bl == 0:
                bl = [x, y]
            if id == 3 and tr == 0:
                tr = [x, y]
            if id == 2 and tl == 0:
                tl = [x, y]
            if id == 1 and br == 0:
                br = [x, y]    
        return tl, tr,br,bl
    
    def onnx_detect(self, img_path):
        try:
            img_bytes_io = io.BytesIO(img_path.read())
            img = Image.open(img_bytes_io)
            img = np.asarray(img)

            tl, tr, br,bl = self.detect_corners(img)
            keypoints = np.zeros((4, 2), dtype = "float32")
            keypoints[0],keypoints[1], keypoints[2],keypoints[3] = tl, tr, br,bl

            wrap = perspective_img(img, keypoints)

            img_wrap = cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB)
            img_result = io.BytesIO()
            img_wrap = cv2.cvtColor(img_wrap, cv2.COLOR_RGB2BGR)  # Thêm dòng này để chuyển từ RGB sang BGR
            cv2.imwrite('output1.png', img_wrap)
            Image.fromarray(img_wrap).save(img_result, format="PNG")
            return True, img_result
        except Exception as e:
            logger.error(e)
            return False, None


# if __name__ == '__main__':
#     model = Detect_Allign()
#     img = Image.open('CMT_test.jpg')
#     img_receive = io.BytesIO()
#     img.save(img_receive, format="PNG")
#     print(type(img_receive))
#     model.onnx_detect(img_receive)