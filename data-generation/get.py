import os
import numpy as np
import cv2
from PIL import Image

backpath='./640,480'
rgbpath='./rgb'
gtpath_instance = './gt_instance'
gtpath_class = './gt_affordance'

outpath1 ='./dataset-generation/JPEGImages'
outpath2 ='./dataset-generation/SegmentationInstance'
outpath3 ='./dataset-generation/SegmentationAffordance'

def get_pascal_labels():

    return np.asarray([[0, 0, 0],  # class 0 'background'  black
                       [255, 128, 0],  # class 1 'cup'       red
                       [255, 255, 128],  # class 2 'coffee'         yellow
                       [0, 255, 128],  # class 3 'cola'       green
                       [128, 255, 255],  # class 4 'knife'     sky blue
                       [128, 128, 255],  # class 5 'bowl'       blue
                       [255, 128, 255],  # class 6 'scissors'     purple
                       [255,0, 255], # hammer
                       [255, 128 ,128],#scoop
                       [244,244,190],
			[180,230,140]]) # turner


def encode_segmap(mask):

    mask = mask.astype(np.uint8)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.uint8)
    return label_mask

file_cls = os.listdir(backpath)
print(file_cls)
backimg = []
for i in range(len(file_cls)):
    backimg_file= os.listdir(backpath+'/'+file_cls[i])
    for f in backimg_file:
        backimg.append(backpath+'/'+file_cls[i]+'/'+f)

print(backimg)

angles=90

if not os.path.exists(outpath1):
    os.mkdir(outpath1)
if not os.path.exists(outpath2):
    os.mkdir(outpath2)
if not os.path.exists(outpath3):
    os.mkdir(outpath3)
i = 0
for rgbname in os.listdir(rgbpath):

    for j in range(0, angles):
        i= i+1
        print(i)

        angle = int(j * 360 / angles)

        img_src = os.path.join(rgbpath, rgbname)
        print(img_src)
        gtname = rgbname[:-4] + '.png'
        mask_instance = os.path.join(gtpath_instance, gtname)
        mask_src_class = os.path.join(gtpath_class, gtname)

        img_src = cv2.imread(img_src)
        h, w, channels = img_src.shape

        backname = np.random.choice(backimg)
        print(backname)
        # back = os.path.join(backpath, backname)
        back = cv2.imread(backname)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

        # clsname=rgbname[0:rgbname.index('_')]

        mask_src1 = cv2.imread(mask_instance)
        mask_main = cv2.warpAffine(mask_src1, M, (w, h))
        outmaskname = outpath2 + '/' + rgbname[:-4] + '_' + str(angle) + '.png'
        cv2.imwrite(outmaskname, mask_main)

        mask_src_class1 = cv2.imread(mask_src_class)
        mask_main_affordance = cv2.warpAffine(mask_src_class1, M, (w, h))
        outmaskname2 = outpath3 + '/' + rgbname[:-4] + '_' + str(angle) + '.png'
        cv2.imwrite(outmaskname2, mask_main_affordance)


        mask_instance = np.asarray(Image.open(mask_instance))
        mask_instance = encode_segmap(mask_instance)
        mask = np.asarray(mask_instance, dtype=np.uint8)

        maskimg_rotate = cv2.warpAffine(mask, M, (w, h))
        sourceimg_rotate = cv2.warpAffine(img_src, M, (w, h))

        sub_img01 = cv2.add(sourceimg_rotate, np.zeros(np.shape(sourceimg_rotate), dtype=np.uint8), mask=maskimg_rotate)

        mask_02 = maskimg_rotate
        mask_02 = np.asarray(mask_02, dtype=np.uint8)

        sub_img02 = cv2.add(back, np.zeros(np.shape(back), dtype=np.uint8),mask=mask_02)
        img_main = back- sub_img02+sub_img01

        outimgname = outpath1 + '/' + rgbname[:-4] +'_'+str(angle) +'.jpg'
        cv2.imwrite(outimgname, img_main)


