import cv2
from imgaug import augmenters as iaa
import os

seq = iaa.Sequential([
    iaa.SomeOf((1, 2),
               [
                   iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),
                   # iaa.Clouds(),
                   iaa.imgcorruptlike.Fog(severity=1),
                   # iaa.imgcorruptlike.Snow(severity=2),
                   iaa.Rain(drop_size=(0.10, 0.15), speed=(0.1, 0.2)),
                   iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.03)),
                   # iaa.FastSnowyLandscape(lightness_threshold=(100, 255),lightness_multiplier=(1.5, 2.0)),
                   # iaa.imgcorruptlike.Spatter(severity=5),
                   iaa.SomeOf((1, 1),
                              [
                                  iaa.imgaug.augmenters.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
                                  iaa.imgcorruptlike.Brightness(severity=(1, 2)),
                                  iaa.imgcorruptlike.Saturate(severity=(1, 3)),
                                  iaa.Multiply((0.3, 0.3)),
                              ]
                              )
               ],
               random_order=True
               )
], random_order=True)

path = ""
savedpath = ""

imglist = []
filelist = os.listdir(path)

for item in filelist:
    img = cv2.imread(path + item)
    # print('item is ',item)
    # print('img is ',img)
    # images = load_batch(batch_idx)
    imglist.append(img)
# print('imglist is ' ,imglist)
print('all the picture have been appent to imglist')

for count in range(1):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename = str(filelist[index])
        cv2.imwrite(savedpath + filename, images_aug[index])
        print('image of count%s index%s has been writen' % (count, index))
