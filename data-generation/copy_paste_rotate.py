"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import imgviz
import cv2
import argparse
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import skimage.io as io
import random




def get_pascal_labels_affordance():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0],  # class 0 'background'  black
                       [255, 0, 0],  # class 1 'grasp'       red
                       [255, 255, 0],  # class 2 'cut'         yellow
                       [0, 255, 0],  # class 3 'scoop'       green
                       [0, 255, 255],  # class 4 'contain'     sky blue
                       [0, 0, 255],  # class 5 'pound'       blue
                       [255, 0, 255],  # class 6 'support'     purple
                       [255, 255, 255],# class wrap-grasp
                       [128,0,128]]) # open

def get_pascal_labels_instance():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0],  # class 0 'background'  black
                       [255, 128, 0],  # class 1 'cup'       red
                       [255, 255, 128],  # class 2 'coffee'         yellow
                       [0, 255, 128],  # class 3 'cola'       green
                       [128, 255, 255],  # class 4 'knife'     sky blue
                       [128, 128, 255],  # class 5 'bowl'       blue
                       [255, 128, 255],  # class 6 'scissors'     purple
                       [255,0, 255], # hammer
                       [255, 128 ,128],#scoop
                       [244,244,190]]) # turner



def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 10
    label_colours = get_pascal_labels_instance()


    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_segmap_affordance(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 9
    label_colours = get_pascal_labels_affordance()


    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask_instance, mask_affordance,e):
    """Encode segmentation label images as pascal classes
    Args:
        mask_instance (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask_instance = mask_instance.astype(np.uint8)
    label_mask_instance = np.zeros((mask_instance.shape[0], mask_instance.shape[1]), dtype=np.uint8)

    if e == 1:
        for ii, label in enumerate(get_pascal_labels_instance()):
            if (label == [0, 0, 0]).all():
                label_mask_instance[np.where(np.all(mask_instance == [0, 0, 0], axis=-1))[:2]] = 0
            else:
                my_instance_class = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                my_instance_class.remove(ii)
                label_mask_instance[np.where(np.all(mask_instance == label, axis=-1))[:2]] = random.choice(
                    my_instance_class)
    if e == 0:
        for ii, label in enumerate(get_pascal_labels_instance()):
            label_mask_instance[np.where(np.all(mask_instance == label, axis=-1))[:2]] = ii





    mask_affordance = mask_affordance.astype(np.uint8)
    label_mask_affordance = np.zeros((mask_affordance.shape[0], mask_affordance.shape[1]), dtype=np.uint8)
    for ii2, label2 in enumerate(get_pascal_labels_affordance()):
        label_mask_affordance[np.where(np.all(mask_affordance == label2, axis=-1))[:2]] = ii2
    label_mask_affordance = label_mask_affordance.astype(np.uint8)
    return label_mask_instance, label_mask_affordance


def random_flip_horizontal(mask, img, p=0.5):

    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    # print('mask:',mask.shape)
    #根据mask模板不为0的元素，输出image中对应位置的像素。
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask) #求图像中的物品，img_src 是原图像，np.zeros创建一个同样大少的图片，

    pic1 = Image.fromarray(np.uint8(mask_src))
    cv2_img1 = cv2.cvtColor(np.asarray(pic1), cv2.COLOR_RGB2BGR)
    cv2.imwrite('test1.jpg',cv2_img1)

    xy1 = get_res_list('test1.jpg')


    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)

    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_mainx = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_mainx,xy1


def img_add3(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)

    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)

    pic2 = Image.fromarray(np.uint8(img_main))
    cv2_img2 = cv2.cvtColor(np.asarray(pic2), cv2.COLOR_RGB2BGR)
    cv2.imwrite('test2.jpg', cv2_img2)

    xy2 = get_res_list('test2.jpg')



    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_mainx = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_mainx,xy2


def img_add2(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask) #求图像中的物品，img_src 是原图像，np.zeros创建一个同样大少的图片，

    pic1 = Image.fromarray(np.uint8(mask_src))
    # cv2_img1 = cv2.cvtColor(np.asarray(pic1), cv2.COLOR_RGB2BGR)


    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_mainx = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_mainx



def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=1.5):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    print("x:",x)
    print('y:', y)

    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask   # mask1
        return mask_pad, img_pad, rescale_ratio, x, y
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop, rescale_ratio, x, y

def Large_Scale_Jittering2(mask, img, rescale_ratio, x, y):

    h, w, _ = img.shape

    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask   # mask1
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop

def copy_paste(mask_src_instance, mask_src_affordance, img_src, mask_main_instance, mask_main_affordance, img_main):

    if args.lsj:
        mask_src_instance, img_src, rescale_ratio, x, y = Large_Scale_Jittering(mask_src_instance, img_src) # mask2
        mask_src_affordance, _ = Large_Scale_Jittering2(mask_src_affordance, img_src, rescale_ratio, x, y)

        mask_main_instance, img_main,  rescale_ratio2, x2, y2 = Large_Scale_Jittering(mask_main_instance, img_main)
        mask_main_affordance, _ = Large_Scale_Jittering2(mask_main_affordance, img_main, rescale_ratio2, x2, y2)

    img, xy1= img_add(img_src, img_main, mask_src_instance)
    mask_instance, xy2 = img_add3(mask_src_instance, mask_main_instance, mask_src_instance)
    mask_affordance = img_add2(mask_src_affordance, mask_main_affordance, mask_src_affordance)
    return mask_instance, img, xy1, xy2, mask_affordance

set=[]
def is_row_exist_object(image, y, xx):

    for x in range(0, xx):
        if image[y, x] != 0:
            return True
    return False
def is_col_exist_object(image, x, low_bound, high_bound):
    for y in range(low_bound, high_bound + 1):
        if image[y, x] != 0:
            return True
    return False
def get_res_list(fliename):
    image = cv2.imread(fliename, cv2.IMREAD_GRAYSCALE)
    yy, xx = image.shape
    res_list = []

    for up_y in range(0, yy):

        if is_row_exist_object(image, up_y, xx):
            res_list.append(up_y)
            break


    for y in range(0, yy):
        down_y = yy - y - 1

        if is_row_exist_object(image, down_y, xx):
            res_list.append(down_y)
            break

    if res_list != []:
        for left_x in range(0, xx):
            if is_col_exist_object(image, left_x, res_list[0], res_list[1]):
                res_list.append(left_x)
                break


        for x in range(0, xx):

            right_x = xx - x - 1
            if is_col_exist_object(image, right_x, res_list[0], res_list[1]):
                res_list.append(right_x)


        # res=(xmin, ymin), (xmax, ymax)
        res = [res_list[2], res_list[0], res_list[3], res_list[1]]
        return res

def write_xml(demoname,clsname,xml_name,a,b,c,d):
    tree = ET.parse(demoname)
    root = tree.getroot()
    object=ET.SubElement(root,"object")
    name=ET.SubElement(object,"name")
    name.text=clsname
    pose=ET.SubElement(object,"pose")
    pose.text="Unspecified"
    truncated=ET.SubElement(object,"truncated")
    truncated.text="0"
    difficult=ET.SubElement(object,"difficult")
    difficult.text="0"
    bndbox=ET.SubElement(object,"bndbox")
    xmin=ET.SubElement(bndbox,"xmin")
    xmin.text=a
    ymin=ET.SubElement(bndbox,"ymin")
    ymin.text=b
    xmax=ET.SubElement(bndbox,"xmax")
    xmax.text=c
    ymax=ET.SubElement(bndbox,"ymax")
    ymax.text=d
    ET.dump(root)
    tree.write(xml_name+'.xml')

def main(args, d):
    # input path
    segInstance = os.path.join(args.input_dir, 'SegmentationInstance')
    segAffordance = os.path.join(args.input_dir, 'SegmentationAffordance')
    JPEGs = os.path.join(args.input_dir, 'JPEGImages')

    # create output path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'SegmentationInstance'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'SegmentatonAffordance'), exist_ok=True)

    xmlpath = os.path.join(args.output_dir, 'Annotations')
    masks_instance_path1 = os.listdir(segInstance)

    tbar = tqdm.tqdm(masks_instance_path1, ncols=100)

    for mask_instance_path in tbar:
        print(d)
        e = 0
        # get source mask and img
        mask_src_instance = np.asarray(Image.open(os.path.join(segInstance, mask_instance_path)), dtype=np.uint8) #ground
        # print('segInstance:', os.path.join(segInstance, mask_instance_path))
        mask_src_affordance = np.asarray(Image.open(os.path.join(segAffordance, mask_instance_path)), dtype=np.uint8)
        # print('mask_src_affordance:', os.path.join(segAffordance, mask_instance_path))
        img_src = cv2.imread(os.path.join(JPEGs, mask_instance_path.replace('.png', '.jpg'))) # rgb
        mask_src_instance, mask_src_affordance = encode_segmap(mask_src_instance, mask_src_affordance,e)

        # random choice main mask/img
        mask_main_path1 = np.random.choice(masks_instance_path1)
        mask_main_instance = np.asarray(Image.open(os.path.join(segInstance, mask_main_path1)),dtype=np.uint8)

        mask_main_affordance = np.asarray(Image.open(os.path.join(segAffordance, mask_main_path1)), dtype=np.uint8)
        img_main = cv2.imread(os.path.join(JPEGs, mask_main_path1.replace('.png', '.jpg')))

        if mask_instance_path.split('_')[0] == mask_main_path1.split('_')[0]:
            e = 1
        mask_main_instance, mask_main_affordance = encode_segmap(mask_main_instance, mask_main_affordance,e)


        # Copy-Paste data augmentation
        try:
            mask, img, xy1, xy2, mask_affordance = copy_paste(mask_src_instance, mask_src_affordance, img_src,
                                                          mask_main_instance, mask_main_affordance, img_main)   # mask
            mask_filename = mask_instance_path
            print(mask_filename)
            img_filename = mask_filename.replace('.png', '.jpg')

            Xmax = max(xy1[0], xy2[0])
            Ymax = max(xy1[1], xy2[1])
            Xmin = min(xy1[2], xy2[2])
            Ymin = min(xy1[3], xy2[3])
            if (Xmax-Xmin)*(Ymax-Ymin) ==(xy1[0]-xy1[2])*(xy1[1]-xy1[3]) or  (Xmax-Xmin)*(Ymax-Ymin) ==(xy2[0]-xy2[2])*(xy2[1]-xy2[3]):
                continue

            rgb = decode_segmap(mask)
            io.imsave(
                os.path.join(args.output_dir, 'SegmentationInstance', mask_filename.replace('.png', '%d.png') % d),
                np.uint8(rgb))

            rgb2 = decode_segmap_affordance(mask_affordance)
            io.imsave(
                os.path.join(args.output_dir, 'SegmentatonAffordance', mask_filename.replace('.png', '%d.png') % d),
                np.uint8(rgb2))

            xmlname = os.path.join(xmlpath, img_filename[:-4] + '%d' % d)
            # xmlname = os.path.join(xmlpath, img_filename[:-4] )
            mask_name1 = os.path.join(segInstance, mask_instance_path)
            # print('name1:', mask_name1)
            clsname1 = mask_instance_path[0:mask_instance_path.index('_')]
            # xy1 = get_res_list(mask_name1)
            write_xml('demo.xml', clsname1, xmlname, str(xy1[0]), str(xy1[1]), str(xy1[2]), str(xy1[3]))

            mask_name2 = os.path.join(segInstance, mask_main_path1)
            # print('name2:', mask_name2)
            clsname2 = mask_main_path1[0:mask_main_path1.index('_')]
            # xy2 = get_res_list(mask_name2)
            write_xml(xmlname + '.xml', clsname2, xmlname, str(xy2[0]), str(xy2[1]), str(xy2[2]), str(xy2[3]))
            cv2.imwrite(os.path.join(args.output_dir, 'JPEGImages', img_filename.replace('.jpg', '%d.jpg') % d), img)
        except Exception as e:
            pass
        continue


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/wzl/TPAMI-code/copy-paste-summary/dataset-generation", type=str,
                        help="input annotated directory")
    parser.add_argument("--output_dir", default="/home/wzl/TPAMI-code/copy-paste-summary/datasets", type=str,
                        help="output dataset directory")
    parser.add_argument("--lsj", default=True, type=bool, help="if use Large Scale Jittering")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for d in range(2):

        main(args,d)

