import pyclipper
import numpy as np
from shapely.geometry import Polygon
from skimage.transform import rotate

def non_max_suppression_fast(boxes, overlapThresh):
    min_score = 0
    east_min_iou = 5
    min_iou = 0.5
    if (len(boxes)==0):
        return [], boxes
    order = np.lexsort((tuple(boxes[:, -2]), tuple(boxes[:, -1])))
    new_boxes = boxes.copy()
    # initialize the list of picked indexes
    pick = []
    suppressed = np.zeros((len(boxes)), dtype=np.int)
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    polygon = list()
    area = list()
    for box in boxes:
        p = list()
        for i in range(4):
            p.append([box[i*2], box[i*2+1]])
        polygon.append(pyclipper.scale_to_clipper(p))
        area.append((((box[0]-box[2])**2+(box[1]-box[3])**2)**0.5)*(((box[2]-box[4])**2+(box[3]-box[5])**2)**0.5))
    order = boxes[:,8].argsort()[::-1]
    for _i in range(len(boxes)):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        pick.append(i)
        area_i = area[i]
        p_i = polygon[i]
        neighbor_list = []
        for _j in range(_i + 1, len(boxes)):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            area_j = area[j]
            p_j = polygon[j]
            try:
                pc = pyclipper.Pyclipper()
                pc.AddPath(p_i, pyclipper.PT_CLIP, True)
                pc.AddPaths([p_j], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION)
                if len(solution) == 0:
                    inter = 0
                else:
                    inter = pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(pyclipper.Area(solution[0])))
            except:
                inter = 0
            iou = inter / (area_i + area_j - inter)
            if iou > min_iou:
                neighbor_list.append(j)
            if (area_i + area_j - inter) > 0 and iou > overlapThresh:
                suppressed[j] = 1
        score_sum = boxes[i][8] - min_score
        for k in range(8):
            new_boxes[i][k] *= score_sum
        for neighbor in neighbor_list:
            for k in range(8):
                new_boxes[i][k] += boxes[neighbor][k] * (boxes[neighbor][8] - min_score)
            score_sum += (boxes[neighbor][8] - min_score)
        for k in range(8):
            new_boxes[i][k] /= score_sum
    return pick, new_boxes