import sys
sys.path.insert(0, './pylayer')
sys.path.insert(0, './caffe/python')
from tool import is_image, load_dict, vec2word, build_voc, write2txt_icdar15_e2e, contain_num, contain_symbol,\
    non_max_suppression_fast
import matplotlib.pyplot as plt
import argparse
import os
import caffe
import numpy as np
import cv2
import cfg
import editdistance


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='test textspotter')
    parser.add_argument(
        '--weight',
        dest='weight',
        default='./models/textspotter.caffemodel',
        type=str)
    parser.add_argument(
        '--prototxt-iou',
        dest='prototxt_iou',
        default='./models/test_iou.pt',
        type=str)
    parser.add_argument(
        '--prototxt-lstm',
        dest='prototxt_lstm',
        default='./models/test_lstm.pt',
        type=str)

    parser.add_argument(
        '--img',
        dest='img',
        default='./imgs/img_105.jpg',
        type=str
    )
    parser.add_argument(
        '--thresholds-ms',
        dest='thresholds',
        default='0.95, 0.95, 0.95, 0.95',
        type=str
    )
    parser.add_argument(
        '--scales',
        dest='scales',
        default='2240, 1920, 1792, 2080',
        type=str
    )

    parser.add_argument(
        '--nms',
        dest='nms',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--rf',
        dest='rf',
        default=10,
        type=int
    )

    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        default='./results',
        type=str
    )




    args = parser.parse_args()
    return args



def predict_single(net, input_fea, previous_word):
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([[cont]])
    word_input = np.array([[previous_word]])
    net.blobs['sample_gt_cont'].reshape(*cont_input.shape)
    net.blobs['sample_gt_cont'].data[...] = cont_input
    net.blobs['sample_gt_label_input'].reshape(*word_input.shape)
    net.blobs['sample_gt_label_input'].data[...] = word_input
    net.blobs['decoder'].reshape(*input_fea.shape)
    net.blobs['decoder'].data[...] = input_fea
    net.forward()
    #net.forward(cont_sel=cont_input, input_sel=word_input, sel_features=input_fea)
    output_preds = net.blobs['probs'].data[0, 0, :]
    return output_preds

def predict_single_from_all_previous(net_lstm, descriptor, previous_words):
    for index, word in enumerate([0] + previous_words):
        res_prob = predict_single(net_lstm, descriptor[[index]], word)
    return res_prob




def forward_iou(im, net_iou, resize_length, mask_th):
    h, w, c = im.shape
    scale = max(h, w) / float(resize_length)

    image_resize_height = int(round(h / scale / 32) * 32)
    image_resize_width = int(round(w / scale / 32) * 32)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height))
    im = np.asarray(im, dtype=np.float32)
    im = im - cfg.mean_val
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, :]

    net_iou.blobs['data'].reshape(*im.shape)
    net_iou.blobs['data'].data[...] = im


    fcn_th_blob = np.zeros((1, 1), dtype=np.float32)
    fcn_th_blob[0, 0] = mask_th
    net_iou.blobs['fcn_th'].reshape(*fcn_th_blob.shape)
    net_iou.blobs['fcn_th'].data[...] = fcn_th_blob
    net_iou.forward()

    det_bboxes = net_iou.blobs['rois'].data[:, 1:].copy()

    det_bboxes[:, :8:2] = det_bboxes[:, :8:2] * scale_w
    det_bboxes[:, 1:8:2] = det_bboxes[:, 1:8:2] * scale_h

    decoder_reg = net_iou.blobs['decoder'].data
    return det_bboxes, decoder_reg



def forward_reg(decoder_rec, net_rec, det_bboxes, recog_th=0.85):
    boxes = list()
    words = list()
    words_score = list()
    det_num = det_bboxes.shape[0]
    if not (det_bboxes > 0).any():
        det_num = 0
    for i in range(det_num):
        previous_words = []
        score = []
        if not (det_bboxes[i] > 0).any():
            continue
        for t in range(cfg.max_len):
            input_fea = decoder_rec[:t + 1, i, :]
            input_fea = np.reshape(input_fea, (t + 1, 1, -1))
            net_rec.blobs['sample_gt_cont'].reshape(1, 1)
            net_rec.blobs['sample_gt_label_input'].reshape(1, 1)
            net_rec.blobs['decoder'].reshape(*input_fea.shape)
            res_probs = predict_single_from_all_previous(net_rec, input_fea, previous_words)
            ind = np.argmax(res_probs)

            if ind == 0:
                break
            else:
                previous_words.append(ind)
                score.append(res_probs[ind])

        if len(score) > 0:
            print float(sum(score)) / len(score), vec2word(previous_words, dicts)
            if float(sum(score)) / len(score) < recog_th:
                continue
            tmp = det_bboxes[i].copy()
            # tmp[-1]+=float(sum(score)) / len(score) * 2
            boxes.append(tmp)
            words.append(vec2word(previous_words, dicts))
            words_score.append(float(sum(score)) / len(score))

    return boxes, words, words_score







if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    if not os.path.exists(args.prototxt_iou) or \
        not os.path.exists(args.prototxt_lstm) or \
        not os.path.exists(args.weight):
        assert False, 'please put model and prototxts in ./model/'

    imgs_files = []
    if os.path.isdir(args.img):
        imgs = os.listdir(args.img)
        imgs_files = [_ for _ in imgs if is_image(_)]
    elif os.path.isfile(args.img):
        imgs = [args.img]
        imgs_files = [_ for _ in imgs if is_image(_)]
    else:
        assert False, 'invalid input image (folder)'

    if len(imgs_files) == 0:
        assert False, 'invalid input image (folder)'

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    caffe.set_mode_gpu()
    caffe.set_device(0)

    net_iou = caffe.Net(args.prototxt_iou, args.weight, caffe.TEST)
    net_rec = caffe.Net(args.prototxt_lstm, args.weight, caffe.TEST)

    thresholds = [float(_) for _ in args.thresholds.strip().split(',')]
    scales = [int(_) for _ in args.scales.strip().split(',')]

    assert len(thresholds) == len(scales), \
        'the length of thresholds and scales should be equal'

    ### get vocabulary
    generic_voc_file = './dicts/generic_lex.txt'
    generic_vocs = load_dict(generic_voc_file)

    dicts = build_voc('./dicts/dict.txt')

    ### forward single image
    for ind, image_name in enumerate(imgs_files):
        new_boxes = list()
        words = list()
        words_score = list()
        image_id = image_name.split('/')[-1].split('.')[0]
        print '%d / %d: ' % (ind+1, len(imgs_files)), image_name
        im = cv2.imread(image_name)
        h, w, c = im.shape
        ### forward every scale
        for k in range(len(scales)):
            image_resize_length = scales[k]
            mask_threshold = thresholds[k]
            det_bboxes, decoder_rec = forward_iou(im, net_iou, image_resize_length, mask_threshold)
            det_num = det_bboxes.shape[0]
            new_boxes, words, words_score = forward_reg(decoder_rec, net_rec, det_bboxes, cfg.recog_th)

        if len(new_boxes) == 0:
            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            new_boxes = np.zeros((0, 8))
            words = np.zeros((0, 8))
            write2txt_icdar15_e2e(out_name, new_boxes, words)
        if len(new_boxes) > 0:
            new_boxes = np.array(new_boxes)
            words = np.array(words)
            words_score = np.array(words_score)
            assert new_boxes.shape[1] == 9
            assert len(new_boxes) == len(words)
            assert len(new_boxes) == len(words_score)

            final_box = list()
            final_words = list()
            final_words_score = list()
            for n in range(new_boxes.shape[0]):
                word = words[n]
                if len(word) < 3:
                    continue
                if (contain_num(word) or contain_symbol(word)) and words_score[n] > cfg.word_score:
                    final_box.append(new_boxes[n])
                    final_words.append(words[n])
                    final_words_score.append(words_score[n])
                    # symbol_or_num = 1
                    continue

                distance = list()
                score = words_score[n]
                for cell in generic_vocs:
                    # dist = levenshteinDistance(word.upper(), cell.upper())
                    dist = editdistance.eval(word.upper(), cell.upper())
                    distance.append(dist)
                    if dist == 0 and words_score[n] > 0.85:
                        score = 1.1
                        #    break
                ind = int(np.argmin(np.array(distance)))
                if (distance[ind] > 1 or score < 0.9):
                    continue
                # if (distance[ind] > 3 or score < 0.9) and has_symbol==1:
                #	continue
                final_box.append(new_boxes[n])
                final_words.append(generic_vocs[ind])
                final_words_score.append(score)

            final_box = np.array(final_box).reshape(-1, 9)
            final_words = np.array(final_words)
            final_words_score = np.array(final_words_score)
            final_box[:, -1] = 2 * final_box[:, -1] + final_words_score
            keep_indices, temp_boxes = non_max_suppression_fast(final_box, args.nms)
            keep_indices = np.int32(keep_indices)

            temp_boxes = final_box[keep_indices]
            temp_words = final_words[keep_indices]

            for index in range(len(keep_indices)):
                for i in range(4):
                    temp_boxes[index][2 * i] = int(round(temp_boxes[index][2 * i]))
                    temp_boxes[index][2 * i] = max(0, temp_boxes[index][2 * i])
                    temp_boxes[index][2 * i] = min(w - 1, temp_boxes[index][2 * i])
                    temp_boxes[index][2 * i + 1] = int(round(temp_boxes[index][2 * i + 1]))
                    temp_boxes[index][2 * i + 1] = max(0, temp_boxes[index][2 * i + 1])
                    temp_boxes[index][2 * i + 1] = min(h - 1, temp_boxes[index][2 * i + 1])

            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            write2txt_icdar15_e2e(out_name, temp_boxes, temp_words)

            ### show results
            plt.imshow(im)
            # print image_resize_width, image_resize_height
            currentAxis = plt.gca()
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            for n in range(len(temp_boxes)):
                coords = np.reshape(temp_boxes[n, 0:8], (-1, 2))
                currentAxis.add_patch(plt.Polygon(coords, fill=False, edgecolor=colors[0], linewidth=2))
                currentAxis.text(coords[0][0], coords[0][1], temp_words[n],
                                 bbox={'facecolor': (1, 0, 0), 'alpha': 0.5})

            plt.show()





