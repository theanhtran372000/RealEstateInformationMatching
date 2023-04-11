import numpy as np

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