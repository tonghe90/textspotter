
from math import cos, sin
import math
import os
import pyclipper
import numpy as np

from math import atan


def rotate(xy, cxcy, theta):
    return (
        cos(theta) * (xy[0] - cxcy[0]) - sin(theta) * (xy[1] - cxcy[1]) + cxcy[0],
        sin(theta) * (xy[0] - cxcy[0]) + cos(theta) * (xy[1] - cxcy[1]) + cxcy[1]
    )


def set_blob_diff(blob, diff):
    blob.reshape(*diff.shape)
    blob.diff[...] = diff


def rect2poly(boxes):
    assert(boxes.shape[1] == 4)
    return np.concatenate((boxes[:,[0,1]], boxes[:,[2,1]], boxes[:,[2,3]], boxes[:,[0,3]]), axis=1)


def set_blob_data(blob, data):
    blob.reshape(*data.shape)
    blob.data[...] = data

def build_voc(voc_file):
    vocs = []
    with open(voc_file) as fp:
        for line in fp:
            line = line.strip()
            vocs.append(str(line).lower())
    return vocs

def is_image(file_name):
    ext = os.path.splitext(file_name)[1].lower()[1:]
    return ext == "jpg" or ext == "JPG"

def vec2word(vec, dicts):
    tmp = ''
    for n in vec:
        if int(n-1) >= len(dicts):
            assert False, "never reach here."
        tmp += dicts[int(n-1)]
    return tmp


def word2vec(vec, dicts):
    res=[]
    for w in vec:
        if w not in dicts:
            assert False, "out of dictionary: {}".format(w)
        res.append(dicts.index(w)+1)
    return res


def rotate_rect(x1,y1,x2,y2,degree,center_x,center_y):
    points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points




def non_max_suppression(boxes, overlapThresh):
    min_score = 0
    min_iou = 0.5
    if (len(boxes)==0):
        return [], boxes
    new_boxes = boxes.copy()
    pick = []
    suppressed = np.zeros((len(boxes)), dtype=np.int)

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



def write2txt_icdar15(file, res):
    f = open(file, 'w')
    det_num = res.shape[0]
    #print res
    if det_num == 0:
        f.close()
        return
    for i in range(det_num):
        f.write('%d,%d,%d,%d,%d,%d,%d,%d\r\n' % (res[i,0],res[i,1],res[i,2],res[i,3],res[i,4],res[i,5],res[i,6],res[i,7]))
    f.close()




def poly_to_box2d_rotate(poly):

    assert (len(poly) == 8)
    edge1 = ((poly[0] - poly[2]) ** 2 + (poly[1] - poly[3]) ** 2) ** 0.5
    angle1 = atan((poly[1] - poly[3]) / (poly[0] - poly[2] + 0.00001))
    edge2 = ((poly[0] - poly[6]) ** 2 + (poly[1] - poly[7]) ** 2) ** 0.5
    angle2 = atan((poly[1] - poly[7]) / (poly[0] - poly[6] + 0.00001))
    if abs(angle1) > abs(angle2):
        w = edge2
        h = edge1
        angle = angle2
    else:
        w = edge1
        h = edge2
        angle = angle1
    cx = (poly[0] + poly[4]) / 2
    cy = (poly[1] + poly[5]) / 2
    return np.array([cx, cy, w, h, angle])



def polyes_to_boxes2d_rotate(polyes):
    """
    :param polyes: N*8  4points (x0,y0,x1,y1.....)
    :return: box2d: N*5 cx, cy, w, h, angle
    """
    poly_num = polyes.shape[0]

    boxes2d = np.zeros((poly_num, 5), dtype=np.float32)
    for n in range(poly_num):

        poly = polyes[n]

        boxes2d[n] = poly_to_box2d_rotate(poly)


    return boxes2d



def box2d_to_poly(box):
    """
    box: [cx, cy, w, h, angle]
    """
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    angle = box[4]
    #print "angle", angle
    # xmin = int(cx - w/2.0)
    # xmax = int(cx + w/2.0)
    # ymin = int(cy - h/2.0)
    # ymax = int(cy + h/2.0)
    xmin = cx - w/2.0
    xmax = cx + w/2.0
    ymin = cy - h/2.0
    ymax = cy + h/2.0
    bb = np.zeros(8, np.float32)
    bb[:2] = rotate((xmin, ymin), (cx, cy), angle)
    bb[2:4] = rotate((xmax, ymin), (cx, cy), angle)
    bb[4:6] = rotate((xmax, ymax), (cx, cy), angle)
    bb[6:] = rotate((xmin, ymax), (cx, cy), angle)

    return bb




def load_dict(dict_file):
    with open(dict_file) as f:
        lexs = (f.read().strip().split())
    return lexs

#


def write2txt_icdar15_e2e(file_name, boxes, words):
    f = open(file_name, 'w')
    det_num = boxes.shape[0]
    words_num = len(words)
    assert det_num == words_num
    if det_num == 0:
        f.close()
        return
    for i in range(det_num):
        f.write('%d,%d,%d,%d,%d,%d,%d,%d,%s\r\n' % (boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3],boxes[i,4],boxes[i,5],boxes[i,6],boxes[i,7], words[i]))
    f.close()


def contain_symbol(word):
    length = len(word)
    flag = 0
    for n in range(length):
        w = word[n]
        if ord("a") <= ord(w) <= ord("z"):
            continue
        elif ord("0") <= ord(w) <= ord("9"):
            continue
        else:
            flag = 1
            break
    return flag

def contain_num(word):
    length = len(word)
    flag = 0
    for n in range(length):
        w = word[n]
        if ord("0")<=ord(w)<=ord("9"):
            flag = 1
            break
    return flag





