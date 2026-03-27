import cv2

def preprocess(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image