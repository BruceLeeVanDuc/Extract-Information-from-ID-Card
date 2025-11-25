import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import gradio as gr
import cv2

def ocr(img, name):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)
    
    width, height = img.size
    print(f"📐 Kích thước: {width} x {height}, Tỉ lệ: {width/height:.3f}")
    
    # ✅ Tính scale từ kích thước chuẩn
    fixed_width = 1280
    fixed_height = 720
    
    if name == 'Căn cước công dân gắn chip':
        img_cmt = img.crop((500/fixed_width * width, 290/fixed_height * height,
                            990/fixed_width * width, 350/fixed_height * height))

        img_ten = img.crop((350/fixed_width * width, 390/fixed_height * height,
                            1100/fixed_width * width, 450/fixed_height * height))

        img_birth = img.crop((730/fixed_width * width, 455/fixed_height * height,
                            1200/fixed_width * width, 495/fixed_height * height))
       
        img_gioi_tinh = img.crop((590/fixed_width * width, 500/fixed_height * height,
                            720/fixed_width * width, 540/fixed_height * height))

        img_quoc_tich = img.crop((1040/fixed_width * width, 500/fixed_height * height,
                                1280/fixed_width * width, 540/fixed_height * height))
        
        img_que_quan = img.crop((250/fixed_width * width, 590/fixed_height * height,
                                1280/fixed_width * width, 650/fixed_height * height))

        img_noi_thuong_tru_line1 = img.crop((840/fixed_width * width, 630/fixed_height * height,
                                1280/fixed_width * width, 690/fixed_height * height))

        img_noi_thuong_tru_line2 = img.crop((250/fixed_width * width, 670/fixed_height * height,
                                1280/fixed_width * width, 730/fixed_height * height))

        img_expiry = img.crop((80/fixed_width * width, 620/fixed_height * height,
                                360/fixed_width * width, 710/fixed_height * height))

    elif name == 'Chứng minh nhân dân':
        img_cmt = img.crop((647.1/fixed_width * width, 192/fixed_height * height,
                            1051/fixed_width * width, 240/fixed_height * height))

        img_ten = img.crop((530/fixed_width * width, 267/fixed_height * height,
                            1241/fixed_width * width, 320/fixed_height * height))

        img_birth = img.crop((750/fixed_width * width, 389/fixed_height * height,
                            1052/fixed_width * width, 450/fixed_height * height))
        
        img_gioi_tinh = img.crop((590/fixed_width * width, 500/fixed_height * height,
                            720/fixed_width * width, 540/fixed_height * height))

        img_quoc_tich = img.crop((1040/fixed_width * width, 500/fixed_height * height,
                                1280/fixed_width * width, 540/fixed_height * height))

        img_que_quan = img.crop((736/fixed_width * width, 449/fixed_height * height,
                                1280/fixed_width * width, 520/fixed_height * height))
        
        img_noi_thuong_tru_line1 = img.crop((570/fixed_width * width, 511/fixed_height * height,
                                1280/fixed_width * width, 580/fixed_height * height))
        
        img_noi_thuong_tru_line2 = ""
        img_expiry = ""

    elif name == 'Căn cước công dân':
        img_cmt = img.crop((580/fixed_width * width, 185/fixed_height * height, 
                            1125/fixed_width * width, 250/fixed_height * height))

        img_ten = img.crop((620/fixed_width * width, 277/fixed_height * height,
                            1280/fixed_width * width, 340/fixed_height * height))

        img_birth = img.crop((770/fixed_width * width, 372/fixed_height * height,
                            1015/fixed_width * width, 415/fixed_height * height))
        
        img_gioi_tinh = img.crop((590/fixed_width * width, 500/fixed_height * height,
                            720/fixed_width * width, 540/fixed_height * height))

        img_quoc_tich = img.crop((1040/fixed_width * width, 500/fixed_height * height,
                                1280/fixed_width * width, 540/fixed_height * height))
        
        img_que_quan = img.crop((600/fixed_width * width, 495/fixed_height * height,
                                1280/fixed_width * width, 580/fixed_height * height))

        img_noi_thuong_tru_line1 = ""
        img_noi_thuong_tru_line2 = ""

        img_expiry = img.crop((263/fixed_width * width, 679/fixed_height * height,
                                455/fixed_width * width, 712/fixed_height * height))
    
    else:
        return None

    images = [img_cmt, img_ten, img_birth, img_gioi_tinh, img_quoc_tich, 
              img_que_quan, img_noi_thuong_tru_line1, img_noi_thuong_tru_line2, img_expiry]
    
    results = []
    for i in range(0, len(images)):
        if images[i] != "":
            results.append(detector.predict(images[i]))
        else:
            results.append("")

    id_num = 'ID number: ' + results[0] + "\n"
    name = 'Name: ' + results[1] + '\n'
    dob = 'Dob: ' + results[2] + '\n'
    gioi_tinh = 'Gender: ' + results[3] + '\n'
    quoc_tich = 'Nationality: ' + results[4] + '\n'
    que_quan = 'Place of origin: ' + results[5] + '\n' if results[5] else "Place of origin: N/A\n"
    
    if ' ' not in results[6]:
        results[6] = ""
    else:
        results[6] += ', '
    
    if ' ' not in results[7]:
        results[7] = ""
    
    noi_thuong_tru = 'Place of residence: ' + results[6] + results[7] + '\n'
    expiry = 'Expiry date: ' + results[8] + '\n' if results[8] else "Expiry date: N/A\n"
    
    if img_noi_thuong_tru_line1 != "" and img_noi_thuong_tru_line2 != "":
        img_noi_thuong_tru_line1 = img_noi_thuong_tru_line1.resize((img_noi_thuong_tru_line2.size[0], img_noi_thuong_tru_line2.size[1]))
        img_noi_thuong_tru = cv2.vconcat([np.array(img_noi_thuong_tru_line1), np.array(img_noi_thuong_tru_line2)])
        img_noi_thuong_tru = Image.fromarray(img_noi_thuong_tru)
    elif img_noi_thuong_tru_line1 != "":
        img_noi_thuong_tru = img_noi_thuong_tru_line1
    else:
        img_noi_thuong_tru = img_que_quan
    
    return (img_cmt, id_num, 
            img_ten, name, 
            img_birth, dob, 
            img_gioi_tinh, gioi_tinh, 
            img_quoc_tich, quoc_tich, 
            img_que_quan, que_quan,
            img_noi_thuong_tru, noi_thuong_tru,
            img_expiry, expiry)


if __name__ == '__main__':
    input = gr.inputs.Image(type="pil")
    demo = gr.interface.Interface(fn=ocr, inputs=input, outputs="text")
    demo.launch()