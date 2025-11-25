from PIL import Image
import gra as gr
import requests
import io
import numpy as np
import cv2
api_url = "http://192.168.1.18:5201/classification"
from io import BytesIO
def predict(image):
    # Convert to bytesio
    img_receive = io.BytesIO()
    image.save(img_receive, format="PNG")
    # receive response
    img_receive.seek(0)
    response = requests.post(api_url,data={"id":1},files={"image": img_receive})
    # img_bytes_io = io.BytesIO(response.content)
    
    # result = Image.open(img_bytes_io)
    response = response.json()
    return response["result"]

if __name__ == '__main__':
    # img = PIL.Image.open('test.jpg')
    # predict(img)
    input = gr.inputs.Image(type="pil")
    demo = gr.interface.Interface(fn=predict, inputs=input, outputs="text")
    demo.launch(share=True)