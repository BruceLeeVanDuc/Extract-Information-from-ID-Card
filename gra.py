from PIL import Image
import gradio as gr
import requests
import io
import numpy as np
import cv2
import json
import os
from align_image.ocr.ocr import ocr


api_url_align = "http://localhost:5203/upload"
api_url_classify = "http://localhost:5201/classification"

# Mapping tên loại thẻ từ API sang tên tiếng Việt
CARD_TYPE_MAPPING = {
    'citizen ID with chip_ front': 'Căn cước công dân gắn chip',
    'ID_ front': 'Chứng minh nhân dân',
    'citizen ID_ front': 'Căn cước công dân',
    'others': 'Không xác định',
    'citizen ID_ back': 'Mặt sau (không hỗ trợ)',
    'citizen ID with chip_ back': 'Mặt sau (không hỗ trợ)',
    'ID_ back': 'Mặt sau (không hỗ trợ)'
}


def predict_align(image):
    try:
        img_receive = io.BytesIO()
        image.save(img_receive, format="PNG")
        img_receive.seek(0)
        response = requests.post(api_url_align, data={"id": 20}, files={"image": img_receive}, timeout=30)
        response.raise_for_status()
        img_bytes_io = io.BytesIO(response.content)
        result = Image.open(img_bytes_io)
        return result
    except Exception as e:
        print(f"Lỗi predict_align: {e}")
        return image

def predict_classify(image):
    try:
        img_receive = io.BytesIO()
        image.save(img_receive, format="PNG")
        img_receive.seek(0)
        response = requests.post(api_url_classify, data={"id": 20}, files={"image": img_receive}, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        return response_json["result"]
    except Exception as e:
        print(f"Lỗi predict_classify: {e}")
        return "others"

def create_error_image():
    error_image_path = 'load_error.png'
    if os.path.exists(error_image_path):
        return Image.open(error_image_path)
    else:
        img = Image.new('RGB', (400, 300), color='red')
        return img

def predict(image):
    try:
        img = predict_align(image)
        classify_raw = predict_classify(img)
        
        # Chuyển đổi tên loại thẻ sang tiếng Việt
        classify = CARD_TYPE_MAPPING.get(classify_raw, classify_raw)
        
        invalid_classes = ["Không xác định", "Mặt sau (không hỗ trợ)"]

        if classify not in invalid_classes:
            (img_cmt, id_num, 
             img_ten, name, 
             img_birth, dob, 
             img_gioi_tinh, gioi_tinh, 
             img_quoc_tich, quoc_tich, 
             img_que_quan, que_quan,
             img_noi_thuong_tru, noi_thuong_tru,
             img_expiry, expiry) = ocr(img, classify)
        else:
            img_error = create_error_image()
            img_cmt = img_ten = img_birth = img_gioi_tinh = img_quoc_tich = img_que_quan = img_noi_thuong_tru = img_expiry = img_error
            id_num = name = dob = gioi_tinh = quoc_tich = que_quan = noi_thuong_tru = expiry = "NONE"

        return (
            img, classify, 
            img_cmt, id_num,
            img_ten, name, 
            img_birth, dob,
            img_gioi_tinh, gioi_tinh, 
            img_quoc_tich, quoc_tich,
            img_que_quan, que_quan,
            img_noi_thuong_tru, noi_thuong_tru,
            img_expiry, expiry
        )
    except Exception as e:
        print(f"Lỗi trong predict: {e}")
        img_error = create_error_image()
        return (
            image, "LỖI", 
            img_error, "LỖI",
            img_error, "LỖI", 
            img_error, "LỖI",
            img_error, "LỖI", 
            img_error, "LỖI",
            img_error, "LỖI",
            img_error, "LỖI",
            img_error, "LỖI"
        )


custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 20px !important;
}

/* Main Container */
#main-container {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 24px !important;
    padding: 40px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
}

/* Header */
.header-title {
    text-align: center;
    margin-bottom: 30px;
}

.header-title h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px !important;
}

.header-title p {
    font-size: 16px !important;
    color: #666 !important;
    font-weight: 400 !important;
}

/* Upload Section */
.upload-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 16px !important;
    padding: 30px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3) !important;
}

.upload-section label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 18px !important;
}

/* Image Preview */
.image-preview {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    background: white !important;
}

/* Button */
.process-button button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 16px 48px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.process-button button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5) !important;
}

/* Results Section */
.results-container {
    background: #f8f9fa !important;
    border-radius: 16px !important;
    padding: 30px !important;
    margin-top: 30px !important;
}

.result-card {
    background: white !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    border-left: 4px solid #667eea !important;
    transition: all 0.3s ease !important;
}

.result-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
    transform: translateX(4px) !important;
}

.result-card label {
    color: #667eea !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 8px !important;
}

.result-card textarea {
    background: #f8f9fa !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
    color: #333 !important;
    font-weight: 500 !important;
}

/* Card Type Badge */
.card-type-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
    margin: 20px 0;
}

/* Tabs */
.tab-nav button {
    background: white !important;
    color: #667eea !important;
    border: 2px solid #667eea !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-color: transparent !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #666;
    font-size: 14px;
}
"""



if __name__ == "__main__":
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        
        with gr.Column(elem_id="main-container"):
            # Header
            gr.HTML("""
                <div class="header-title">
                    <h1>🆔 Smart ID Card Scanner</h1>
                    <p>Trích xuất thông tin từ CMND/CCCD tự động với AI</p>
                </div>
            """)
            
            # Upload Section
            with gr.Row():
                with gr.Column(scale=1, elem_classes="upload-section"):
                    input_align = gr.Image(
                        type="pil", 
                        label="📤 Upload ảnh CMND/CCCD",
                        elem_classes="image-preview"
                    )
                    
                    image_align_button = gr.Button(
                        "🚀 Bắt đầu xử lý",
                        size="lg",
                        elem_classes="process-button"
                    )
            
            # Aligned Image Preview
            with gr.Row():
                output_align = gr.Image(
                    type="pil", 
                    label="✅ Ảnh đã căn chỉnh",
                    elem_classes="image-preview"
                )
            
            # Card Type
            classify = gr.Textbox(
                label="🏷️ Loại thẻ",
                interactive=False,
                elem_classes="result-card"
            )
            
            # Results in Tabs
            with gr.Tabs():
                with gr.Tab("📋 Thông tin cơ bản"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                id_num = gr.Textbox(label="🔢 Số CMND/CCCD", interactive=False)
                                img_cmt = gr.Image(type="pil", label="Preview", show_label=False)
                        
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                name = gr.Textbox(label="👤 Họ và tên", interactive=False)
                                img_ten = gr.Image(type="pil", label="Preview", show_label=False)
                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                dob = gr.Textbox(label="🎂 Ngày sinh", interactive=False)
                                img_birth = gr.Image(type="pil", label="Preview", show_label=False)
                        
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                gioi_tinh = gr.Textbox(label="⚧ Giới tính", interactive=False)
                                img_gioi_tinh = gr.Image(type="pil", label="Preview", show_label=False)
                
                with gr.Tab("🌍 Địa chỉ & Quốc tịch"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                quoc_tich = gr.Textbox(label="🏳️ Quốc tịch", interactive=False)
                                img_quoc_tich = gr.Image(type="pil", label="Preview", show_label=False)
                        
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                que_quan = gr.Textbox(label="🏡 Quê quán", interactive=False)
                                img_que_quan = gr.Image(type="pil", label="Preview", show_label=False)
                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                noi_thuong_tru = gr.Textbox(label="📍 Nơi thường trú", interactive=False)
                                img_noi_thuong_tru = gr.Image(type="pil", label="Preview", show_label=False)
                
                with gr.Tab("📅 Thời hạn"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group(elem_classes="result-card"):
                                expiry = gr.Textbox(label="⏰ Ngày hết hạn", interactive=False)
                                img_expiry = gr.Image(type="pil", label="Preview", show_label=False)
            
            # Footer
            gr.HTML("""
                <div class="footer">
                    <p>💡 Made with ❤️ by Team</p>
                </div>
            """)
            
            # Button Click Event
            image_align_button.click(
                predict,
                inputs=input_align,
                outputs=[
                    output_align, classify,
                    img_cmt, id_num,
                    img_ten, name,
                    img_birth, dob,
                    img_gioi_tinh, gioi_tinh,
                    img_quoc_tich, quoc_tich,
                    img_que_quan, que_quan,
                    img_noi_thuong_tru, noi_thuong_tru,
                    img_expiry, expiry
                ]
            )
    
    demo.launch(server_name="0.0.0.0", server_port=7878)