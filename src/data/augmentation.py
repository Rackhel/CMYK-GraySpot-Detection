import cv2
import random
import numpy as np

def augment(image):
    
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    if random.random() > 0.5:
        value = random.randint(-30, 30)
        image = cv2.add(image, value)
        
    if random.random() > 0.5:
        noise = random.randint(0, 10)
        image = np.clip(image + noise, 0, 255)

    return image