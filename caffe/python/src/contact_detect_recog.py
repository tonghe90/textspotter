import os
import cv2

use_dict = True
plot = False

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

detect_path = '/data1/liuxuebo/data/icdar/icdar15_result_test_e2e'
recog_file  = '/data1/liuxuebo/recog/icdar15_train_pred.txt'
contact_out_list = '/data1/liuxuebo/text_detect/contact_result'
dict_path = '/data1/liuxuebo/data/icdar/ch4_test_vocabularies_per_image'
threshold = 0.5
detect_file_list = map(lambda x: os.path.join(detect_path, x), os.listdir(detect_path))
recog_path_array = list()
recog_label_array = list()
with open(recog_file) as fr_recog:
	for line in fr_recog:
		recog_path_array.append(line.split()[0])
		recog_label_array.append(line.split()[1])

for detect_file in detect_file_list:
	if detect_file.endswith('txt'):
		if detect_file.endswith('t.txt'):
			continue
		with open(detect_file) as fr_detect:
			with open('{}/{}'.format(contact_out_list, detect_file.split('/')[-1]), 'w') as fw:
				dict_name = 'voc' + detect_file.split('/')[-1][3:]
				img_name = detect_file[0:-4] + '.jpg'
				print(img_name)
				if plot:
					img = cv2.imread(img_name)
				# print(dict_name)
				with open('{}/{}'.format(dict_path, dict_name)) as fr_dict:
					dict_ = fr_dict.read().strip().split()
					for line in fr_detect:
						x1, y1, x2, y2, x3, y3, x4, y4, path = line.strip().split(',')
						if(path not in recog_path_array):
							continue
						index = recog_path_array.index(path)
						label_tmp = recog_label_array[index]
						if label_tmp.isdigit() or not use_dict:
							label = label_tmp
						else:
							distance = list()
							for cell in dict_:
								distance.append(levenshteinDistance(label_tmp.upper(), cell.upper()))
							# TODO: multi min distance
							min_dis = distance.index(min(distance))
							label = dict_[min_dis]
							# if(len(label_tmp) < 3):
							# 	label = label_tmp
							# float(levenshteinDistance(label_tmp.upper(), label.upper()))
							# print(label, label_tmp, min(distance), levenshteinDistance(label_tmp.upper(), label.upper()))
							if float(min(distance)) / (len(label_tmp)) > threshold:
								continue
						fw.write('{},{},{},{},{},{},{},{},{}\r\n'.format(x1, y1, x2, y2, x3, y3, x4, y4, label))
						if plot:
							cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,0), 2)
							cv2.line(img, (int(x1),int(y1)), (int(x4),int(y4)), (0,0,0), 2)
							cv2.line(img, (int(x3),int(y3)), (int(x2),int(y2)), (0,0,0), 2)
							cv2.line(img, (int(x3),int(y3)), (int(x4),int(y4)), (0,0,0), 2)
							cv2.putText(img, label, (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
							cv2.imwrite('{}/{}'.format(contact_out_list, img_name.split('/')[-1]),img)