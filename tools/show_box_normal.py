import json
import cv2

filename = r'/data/wangyh/mllms//MultimodalOCR/data/docVQA/val/images/documents/nkbl0226_1.png'


_to_save_img = cv2.imread(filename)
h, w, _ = _to_save_img.shape
boxes = [[0.01, 0.05, 0.83, 0.11]]
for i, box_seq in enumerate(boxes):
    boxes[i][0] = int(boxes[i][0] * w)
    boxes[i][1] = int(boxes[i][1] * h)
    boxes[i][2] = int(boxes[i][2] * w)
    boxes[i][3] = int(boxes[i][3] * h)

color_boxes = [(0, 191, 255), (255, 0, 255), (0, 250, 154), (255, 255, 0)]
# bboxes = [[8, 52, 620, 355], [1143, 250, 1219, 414], [645, 295, 741, 332], [558, 375, 844, 423]]
j = 0
for i in boxes:
    # color = (int(color[0]), int(color[1]), int(color[2]))
    # print(color)
    # cv2.rectangle(_to_save_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
    # cv2.rectangle(_to_save_img, (i[0], i[1]), (i[2], i[3]), color, 2)
    cv2.rectangle(_to_save_img, (i[0], i[1]), (i[2], i[3]), color_boxes[j], 5)
    j += 1
cv2.imwrite('bbox.jpg', _to_save_img)
