import requests
from PIL import Image
from io import BytesIO
import numpy as np
import base64

class ImageURL:

    def __init__(self):
        pass

    def image_url_to_array(self, file):
        try:
            #response = requests.get(image_url)
            #response.raise_for_status()  # Raise an error for bad responses
            #image = Image.open(BytesIO(image_data)).convert("RGB")
            image = Image.open(file.file).convert("RGB")
            return np.array(image), None

        except Exception as e:
            print(f"Error downloading or processing the image: {e}")
            return None, str(e)
