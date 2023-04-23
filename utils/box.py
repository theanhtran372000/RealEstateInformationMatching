import cv2
import numpy as np
from .orientation import get_theta

# Padding box
def pad_boxes(boxes, im_w, im_h, w_pad=2, h_pad=2):
    new_boxes = []
    for box in boxes:
        tl = [max(box[0][0] - w_pad, 0), max(box[0][1] - h_pad, 0)]
        tr = [min(box[1][0] + w_pad, im_w - 1), max(box[1][1] - h_pad, 0)]
        br = [min(box[2][0] + w_pad, im_w - 1), min(box[2][1] + h_pad, im_h - 1)]
        bl = [max(box[3][0] - w_pad, 0), min(box[3][1] + h_pad, im_h - 1)]
        
        new_boxes.append([tl, tr, br, bl])
        
    return np.array(new_boxes)

# === Line merge === #

# Euclide distance
def dist(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    
    return np.sqrt(np.sum(np.subtract(pt1, pt2) ** 2))

# box1 is next to box2 (box1 in front)
def next_to(box1, box2, thresh=0.3):
    d1 = dist(box1[1], box2[0])
    d2 = dist(box1[2], box2[3])
    
    max_h = max(box1[3][1] - box1[0][1], box2[3][1] - box2[0][1])
    
    if (d1 + d2) / 2 <= thresh * max_h:
        return True
    else:
        return False

# Get right most element in a group
def right_most_element(group):
    if len(group) == 0:
        return None
    
    rme = group[0]
    
    for e in group[1:]:
        if e[1][0] + e[2][0] > rme[1][0] + rme[2][0]:
            rme = e
            
    return rme

# Sort box from left to right
def sort_boxes(boxes, l2r=True):
    return np.array(sorted(boxes, key=lambda box: box[0][0] + box[3][0], reverse=not l2r))

# Group boxes into line
def group_boxes_v2(boxes, thresh=0.3):
    boxes = sort_boxes(boxes)
    
    groups = []
    
    for box in boxes:
        
        done = False
        for i, group in enumerate(groups):
            rme = right_most_element(group)
            
            if next_to(rme, box, thresh):
                groups[i].append(box)
                done = True
                break
                
        if not done:
            groups.append([box])
            
    return groups  

# Enhance contrast (not used)
def increase_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

# Create polygon box from box list
def create_polygon(group):
    top_pts = []
    bot_pts = []
    
    for box in group:
        top_pts.extend([np.int32(box[0]), np.int32(box[1])])
        bot_pts.extend([np.int32(box[3]), np.int32(box[2])])
        
    top_pts = sorted(top_pts, key=lambda pt: pt[0])
    bot_pts = sorted(bot_pts, key=lambda pt: pt[0], reverse=True)
    
    return top_pts + bot_pts

# Visualize all polygons on image
def draw_polygons(img, lines, color=(0, 0, 255), thickness=1):
    polys = []
    for line in lines:
        polys += [np.int32(result[2]) for result in line]
    return cv2.polylines(img.copy(), polys, isClosed=True, color=color, thickness=thickness)

# Shrink image
def shrink(img, v=0):
    result = img
    h, w, _ = result.shape
    
    # Shrink image
    min_x, min_y = 0, 0
    max_x, max_y = w-1, h-1
    
    for x in range(w):
        if (result[:, x, :] != v).any():
            min_x = x
            break
    
    for x in range(w-1, -1, -1):
        if (result[:, x, :] != v).any():
            max_x = x
            break
            
    for y in range(h):
        if (result[y, :, :] != v).any():
            min_y = y
            break
    
    for y in range(h-1, -1, -1):
        if (result[y, :, :] != v).any():
            max_y = y
            break
    
    center = [
        (max_x + min_x) / 2,
        (max_y + min_y) / 2
    ]
    
    result = result[min_y:max_y, min_x:max_x]
    
    text_h = max_y - min_y
    
    return result, center, text_h

def rotate(image, angle, center=None, scale=1.0, color=(255, 255, 255)):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color
    )

    # return the rotated image
    return rotated

def align_text(text_img):
    
    img = text_img.copy()
    
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary image
    thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Reverse color (text -> white, bg -> black)
    img = 255 - img
    
    # Orientation
    _, _, theta = get_theta(img)
    
    # Rotate image
    text_img = rotate(text_img, theta / np.pi * 180)
    
    return text_img

# Fill edge
def fill_image(img, old_value=0, new_value=255):
    img[img == old_value] = new_value
    return img

# Post process merge image
def postprocess_result(result):
    
    # Shrink image
    result, center, text_h = shrink(result, v=0)
    
    # Fill image
    result = fill_image(result, 0, 255)
    
    # Align result
    result = align_text(result)
    
    # Fill image
    result = fill_image(result, 0, 255)
    
    # Shrink text
    result, _, _ = shrink(result, v=255)
    
#     # Convert to binary
#     result = increase_contrast(result)
#     gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Convert to BGR
#     result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return result, center, text_h

# Extract a box from image
def pattern_mask(image, box):
    if image is not None:
        h, w, _ = image.shape
        if box is not None:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, np.int32(box), 255)
            image = cv2.bitwise_and(image, image, mask=mask)
    return image

# Extract a group of boxes from image
def extract_group(group, img):
    masks = []
    for box in group:
        masks.append(pattern_mask(img, box))
        
    result = masks[0]
    for mask in masks[1:]:
        result = cv2.bitwise_or(result, mask)
        
    # Shrink + Fill background
    result, center, text_h = postprocess_result(result)
    
    # Create polygon
    poly = create_polygon(group)
    
    return result, center, poly, text_h

# Analysis line info
def get_line_info(line):
    mean_center = [
        np.mean([r[1][0] for r in line]),
        np.mean([r[1][1] for r in line])
    ]
    
    mean_h = np.mean([r[3] for r in line])
    
    return mean_center, mean_h

# Check if r1 in line (contains multiple result)
def in_line(re, line, f=0.25):
    thresh = f * re[3]
    
    mean_center, mean_h = get_line_info(line)
    
    if np.abs(mean_center[1] - re[1][1]) <= thresh:
        return True
    else:
        return False

# Sort boxes by center height
def group_to_lines(results, f=0.25, reverse=False):
    
    # Group into lines
    lines = []
    
    for r in results:
        found = False
        
        for i, line in enumerate(lines):
            if in_line(r, line, f):
                lines[i].append(r)
                found = True
                break
                
        if not found:
            lines.append([r])
        
    # Sorted each line
    sorted_lines = []
    
    for line in lines:
        sorted_lines.append(sorted(line, key=lambda r: r[1][0], reverse=reverse))
    
    # Sorted list line
    sorted_lines = sorted(sorted_lines, key=lambda line: get_line_info(line)[0][1], reverse=reverse)
        
    return sorted_lines

# Extract all groups
def extract_groups(groups, img, f=0.25):
    results = []
    for group in groups:
        results.append(extract_group(group, img))
        
    lines = group_to_lines(results, f)
    return lines

# === Horizontal merge === #
def get_info(i, box):
    min_x = min([point[0] for point in box])
    max_x = max([point[0] for point in box])
    min_y = min([point[1] for point in box])
    max_y = max([point[1] for point in box])
    
    w = max_y - min_y
    h = max_x - min_x
    
    return {
        'i': i,
        'min_x': min_x, 
        'max_x': max_x, 
        'min_y': min_y, 
        'max_y': max_y, 
        'w': w, 
        'h': h
    }

def approx(info1, info2, factor=0.25):
    # In same row
    if  info1['min_y'] - factor * info1['h'] <= info2['min_y'] and \
        info2['max_y'] <= info1['max_y'] + factor * info1['h']:
        return True
    else:
        return False

def sort_infos(infos, l2r=True):
    infos = sorted(infos, key=lambda info: info['min_x'], reverse = not l2r)
    return infos

def get_right_most_mem(group):
    max_x = 0
    chosen = None
    
    for mem in group:
        if mem['max_x'] > max_x:
            max_x = mem['max_x']
            chosen = mem
            
    return mem
    
def near(info1, info2, factor=1.):
    if info1['min_x'] <= info2['max_x'] + info2['w'] * factor:
        return True
    return False
    
def get_mean_group(i, group):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    
    w = 0
    h = 0
    
    l = len(group)
    
    for mem in group:
        min_x += mem['min_x']
        max_x += mem['max_x']
        min_y += mem['min_y']
        max_y += mem['max_y']
        
        w += mem['w']
        h += mem['h']
        
    return {
        'i': i,
        'min_x': int(min_x / l), 
        'max_x': int(max_x / l), 
        'min_y': int(min_y / l), 
        'max_y': int(max_y / l), 
        'w': int(w / l), 
        'h': int(h / l)
    }

def process_group(group):
    first_mem = group[0]
    
    min_x = first_mem['min_x']
    max_x = first_mem['max_x']
    min_y = first_mem['min_y']
    max_y = first_mem['max_y']
    
    for mem in group[1:]:
        min_x = min(min_x, mem['min_x'])
        max_x = max(max_x, mem['max_x'])
        min_y = min(min_y, mem['min_y'])
        max_y = max(max_y, mem['max_y'])
        
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

def group_boxes(boxes):
    # Grouping
    groups = []
    infos = []
    
    for i, box in enumerate(boxes):
        # Extract info
        infos.append(get_info(i, box))
    
    # sort box from left to right
    infos = sort_infos(infos)
    
    for i, info in enumerate(infos):
        done = False
        for i, group in enumerate(groups):
            mean_info = get_mean_group(i, group)
            
            right_most_mem = get_right_most_mem(group)
            
            if approx(mean_info, info) and near(info, right_most_mem):
                groups[i].append(info)
                done = True
                break

        if not done:
            groups.append([info])
            
    return groups

def process_group(group):
    first_mem = group[0]
    
    min_x = first_mem['min_x']
    max_x = first_mem['max_x']
    min_y = first_mem['min_y']
    max_y = first_mem['max_y']
    
    for mem in group[1:]:
        min_x = min(min_x, mem['min_x'])
        max_x = max(max_x, mem['max_x'])
        min_y = min(min_y, mem['min_y'])
        max_y = max(max_y, mem['max_y'])
        
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

def sort_boxes_vertical(boxes, h2l=True):
    return sorted(boxes, key=lambda box: box[0][1], reverse=not h2l)

def sort_boxes_horizontal(boxes, l2r=True):
    return sorted(boxes, key=lambda box: box[0][0], reverse=not l2r)

def merge_groups(groups):
    merged_boxes = []
    for group in groups:
        merged_boxes.append(process_group(group))
    
    merged_boxes = sort_boxes_vertical(merged_boxes)
    merged_boxes = np.array(merged_boxes)
    
    return merged_boxes

def get_center(box):
    min_x, min_y = box[0]
    max_x, max_y = box[2]
    
    return [
        int((max_x + min_x) / 2),
        int((max_y + min_y) / 2)
    ]

def merge_lines(boxes):
    
    # Group box into lines
    lines = []
    for box in boxes:
        
        min_x, min_y = box[0]
        max_x, max_y = box[2]
        h = max_y - min_y
        
        found = False
        for i, line in enumerate(lines):
            centers = [get_center(box) for box in line]
            
            mean_center = [
                int(sum([c[0] for c in centers]) / len(centers)),
                int(sum([c[1] for c in centers]) / len(centers))
            ]
            
            if min_y <= mean_center[1] <= max_y:
                lines[i].append(box)
                found = True
                break
                
        if not found:
            lines.append([box])
    
    # Sort lines
    for i, line in enumerate(lines):
        lines[i] = sort_boxes_horizontal(line)
        
    return lines
    
def merge_boxes(boxes):
    groups = group_boxes(boxes)
    boxes = merge_groups(groups)
    lines = merge_lines(boxes)
    return boxes, lines