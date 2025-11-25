import os
import sys
sys.path.append(".")
from server.logger_ser import logger
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
import numpy as np
import io
import cv2


classes = {0: "citizen ID_ back", 1: "citizen ID_ front",
        2: "citizen ID with chip_ front", 3:  "citizen ID with chip_ back"
        , 4: "ID_ back", 5: "ID_ front", 6: "others"}

class Classifier:
    def __init__(self):
        self.session = self.__load_onnx_session()

    def __load_onnx_session(
        self,
        onnx_path="onnx_checkpoints/model_best_mbv2.onnx",
    ):
        provider = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_path, providers=provider)
        return session

    def onnx_classify(self,  img_path, uuid, thresh=0.6):
        try:
            img_bytes_io = io.BytesIO(img_path.read())
            img = Image.open(img_bytes_io)
            print(type(img))
            image = img.copy().resize((224, 224), Image.BILINEAR)
            image = np.array(image)
            cv2.imwrite('output.png', image)
            original_image = image.copy()

            mean = np.array([0, 0, 0])
            std = np.array([255, 255, 255])
            image = (image - mean) / std

            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.expand_dims(image, axis=0)  # CHW -> BCHW
 
            image = np.float32(image)

            out = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]
            out = self.__softmax(out)[0]

            # draw = ImageDraw.Draw(image)
            # text = idx +"_ " + classes[idx] + "score: " + round(100 * out[idx].item(), 2)
            # font = ImageFont.truetype('arial.ttf', 36)
            # draw.text((0, 0), text, font=font)

            idx = int(np.argmax(out))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"/home/thanh/workspace/api-server/server/static/result/image{uuid}_{classes[idx]}.jpg",
            #              original_image)
            if out[idx] > thresh:
                return True, {
                    "type_code": idx,
                    "img_path": img,
                    "class": classes[idx],
                    "score": round(100 * out[idx].item(), 2),
                }
            else:
                return True, {
                    "type_code": idx,
                    "img_path": img,
                    "class": "others",
                    "type_msg": classes[idx],
                }        
            
        except Exception as e:
            logger.error(e)
            return False, None

    # for validating purpose, use this function in validation.py
    def onnx_validating(self, img_path, thresh=0.6):
        if type(img_path) == str:
            image_root = Image.open(img_path).convert("RGB")
        else:
            image_root = Image.open(io.BytesIO(img_path)).convert("RGB")

        image = image_root.copy().resize((512, 512), Image.BILINEAR)
        image = np.array(image)

        mean = np.array([0, 0, 0])
        std = np.array([255, 255, 255])
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # CHW -> BCHW

        image = np.float32(image)

        out = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]
        out = self.__softmax(out)[0]
        return out

    @staticmethod
    def __softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

