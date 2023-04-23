from PIL import Image
import cv2

def save_cv2_img(img, path):
    img = img[:, :, ::-1]
    pil_img = Image.fromarray(img)
    pil_img.save(path)