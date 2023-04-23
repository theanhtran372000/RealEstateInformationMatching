import numpy as np
import math


def bf(image, x, y):
    return 1 if image[y, x] > 0 else 0

def get_area(image):
    return np.sum(image == 255)

def get_center(image, area):
    
    x_sum = 0
    y_sum = 0
    h, w = image.shape
    for x in range(w):
        for y in range(h):
            x_sum += x * bf(image, x, y)
            y_sum += y * bf(image, x, y)
            
    return (int(x_sum / area), int(y_sum / area))

def get_abc(image, center):
    a = 0
    b = 0
    c = 0
    
    x_, y_ = center
    
    h, w = image.shape
    for x in range(w):
        for y in range(h):
            _x = x - x_
            _y = y - y_
            a += _x ** 2 * bf(image, x, y)
            b += 2 * _x * _y  * bf(image, x, y)
            c += _y ** 2 * bf(image, x, y)
            
    return a, b, c

def get_theta(image):
    area = get_area(image)
    center = get_center(image, area)
    a, b, c = get_abc(image, center)
    theta = math.atan(b / (a - c)) / 2
    return area, center, theta

def get_angle(main_theta, side_theta):
    if side_theta > np.pi / 2:
        side_theta -= np.pi
        
    return np.pi / 2 - math.fabs(main_theta - side_theta)
