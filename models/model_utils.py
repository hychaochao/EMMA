import base64
from io import BytesIO

def encode_image_to_base64(image):
    # 将 PIL.Image 转为 Base64 编码的字符串
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 确保保存为 PNG 格式
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str