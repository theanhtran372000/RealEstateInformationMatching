import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def vertical_crop(img):
    # Convert image to grayy scale
    img_cp = img.copy()
    gray_img = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    
    # Detect lines
    canny_image = cv2.Canny(gray_img, 70, 200)
    lines = cv2.HoughLines(canny_image, 1, np.pi/180, 30)
    
    # Filter vertical lines
    vertical_lines = []
    img_w = gray_img.shape[1]
    for line in lines:
        rho, theta = line[0]
        if theta < 0.2 and rho <= int(img_w * 0.6) and rho >= int(img_w * 0.4):
            vertical_lines.append(line)

    vertical_lines = np.array(vertical_lines)
    
    # Crop image
    if len(vertical_lines) > 0:
        x_crop = int(vertical_lines[0][0][0]) # rho of the first line
        l_img, r_img = img_cp[:, :x_crop], img_cp[:, x_crop:]
        return l_img, r_img
    else:
        return img_cp, img_cp
    
def box_crop(img, box):
    
    min_x, min_y = [int(x) for x in box[0]]
    max_x, max_y = [int(x) for x in box[2]]

    subimg = img[min_y:max_y, min_x:max_x]

    return subimg
    
def box_crop_and_save(img, boxes, save_dir):
    
    subimgs = []
    for i, box in enumerate(boxes):
        subimg = box_crop(img, box)
        subimgs.append(subimg)
        
        cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(str(i).zfill(3))), subimg)
        
    return subimgs
    