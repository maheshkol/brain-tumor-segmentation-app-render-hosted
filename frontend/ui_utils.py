import base64
from PIL import Image
from io import BytesIO

def decode_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))
