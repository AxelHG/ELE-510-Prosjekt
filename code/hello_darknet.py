import sys, os
import time
import cv2
sys.path.append(os.path.join(os.getcwd(),'/home/axel/Git/darknet/python/'))
import darknet as dn

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    t = time.time()
    im = array_to_image(image)
    print('array_to_image time: {}').format(time.time() - t)
    dn.rgbgr_image(im)
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    t = time.time()
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    # free_detections(dets, num)
    return res


# Darknet
t_start = time.time()
net = dn.load_net("/home/axel/Git/darknet/cfg/yolov3-spp.cfg", "/home/axel/Git/darknet/yolov3-spp.weights", 0)
meta = dn.load_meta("/home/axel/Git/darknet/cfg/coco.data")
b = detect(net, meta, cv2.imread('/home/axel/Git/darknet/data/dog.jpg'))
print('Time spendt total: {}').format(time.time() - t_start)
print b
