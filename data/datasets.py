import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
import data
import model.models
import utils
import os
import urllib.request
import scipy.io
import zipfile
import requests
import sklearn.utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import color as skimage_color
import skimage.filters
import gc
import glob
import multiprocessing
import math
import random
import shutil

def rgb2lab(r, g, b):

    r /= 255
    g /= 255
    b /= 255

    if (r > 0.04045):
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92

    if (g > 0.04045):
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92

    if (b > 0.04045):
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    if (x > 0.008856):
        x = pow(x, 1 / 3)
    else:
        x = (7.787 * x) + 16 / 116

    if (y > 0.008856):
        y = pow(y, 1 / 3)
    else:
        y = (7.787 * y) + 16 / 116

    if (z > 0.008856):
        z = pow(z, 1 / 3)
    else:
        z = (7.787 * z) + 16 / 116

    l = (116 * y) - 16
    a = 500 * (x - y)
    bb = 200 * (y - z)
    return l, a, bb


def rotate(image, angle, dtype='float32'):

    return np.array(cv2.warpAffine(np.array(image, dtype='float32'),
                                   cv2.getRotationMatrix2D((image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5), angle,
                                                           1.0), (image.shape[0], image.shape[1])), dtype=dtype)


def motion_blur(image, size, angle, dtype='float32'):

    k = np.zeros((size, size), dtype='float32')
    k[(size - 1) // 2, :] = np.ones(size, dtype='float32')
    k = rotate(k, angle)
    k = np.array(k / np.sum(k), dtype='float32')
    image = np.array(image, dtype='float32')
    return np.array(cv2.filter2D(image, -1, k), dtype=dtype)


def occlusion(image, startx, starty, lenx, leny):

    image[startx:startx + lenx, starty:starty + leny, :] = 0


def occlusion_a(aImages, pixels=10, verbose=False):

    imgLen = len(aImages)
    for img in range(imgLen):
        startx = random.randint(0, aImages[img].shape[0] - pixels)
        starty = random.randint(0, aImages[img].shape[1] - pixels)
        occlusion(image=aImages[img], startx=startx, starty=starty, lenx=pixels, leny=pixels)
        if (img % 1000 == 0):
            gc.collect()
            if verbose and (img > 0):
                print(img, ' images processed.')
    gc.collect()


def motion_blur_a(aImages, pixels=10, dtype='float32', verbose=False):

    imgLen = len(aImages)
    for img in range(imgLen):
        aImages[img] = motion_blur(image=aImages[img], size=pixels, angle=random.randint(0, 180), dtype=dtype)
        if (img % 1000 == 0):
            gc.collect()
            if verbose and (img > 0):
                print(img, ' images processed.')
    gc.collect()


def rgb2lab_a(aRGB, verbose=True):

    imgLen = len(aRGB)
    xLen = len(aRGB[0])
    yLen = len(aRGB[0][0])

    for img in range(imgLen):
        if verbose and (img % 1000 == 0):
            print(img, ' images converted to lab.')
        for x in range(xLen):
            for y in range(yLen):
                r = aRGB[img][x][y][0]
                g = aRGB[img][x][y][1]
                b = aRGB[img][x][y][2]
                l, a, bb = rgb2lab(r, g, b)
                aRGB[img][x][y][0] = l
                aRGB[img][x][y][1] = a
                aRGB[img][x][y][2] = bb


def skimage_rgb2lab_a(aRGB, verbose=True):

    imgLen = len(aRGB)
    for img in range(imgLen):
        aRGB[img] = skimage_color.rgb2lab(aRGB[img])
        if (img % 1000 == 0):
            gc.collect()
            if verbose and (img > 0):
                print(img, ' images converted to lab.')
    gc.collect()


def cv2_resize_a(aImages, target_size=(64, 64), interpolation=cv2.INTER_NEAREST, verbose=False):
    imgLen = len(aImages)
    outImg = []
    for img in range(imgLen):
        outImg.append(cv2.resize(aImages[img], dsize=target_size, interpolation=interpolation))
        if (img % 1000 == 0):
            gc.collect()
            if verbose:
                print(img, ' images resized.')
    gc.collect()
    return np.array(outImg)


def skimage_blur(aImages, sigma=1.0, truncate=3.5, verbose=True):
    imgLen = len(aImages)
    for img in range(imgLen):
        aImages[img] = np.array(skimage.filters.gaussian(
            np.array(aImages[img], dtype='float32'),
            sigma=(sigma, sigma), truncate=truncate, multichannel=True), dtype='float32')
        if (img % 1000 == 0):
            gc.collect()
            if verbose:
                print(img, ' images blurred.')
    gc.collect()


def salt_pepper(aImages, salt_pepper_num, salt_value=2.0, pepper_value=-2.0, verbose=True):
    imgLen = len(aImages)
    for img in range(imgLen):
        current_image = aImages[img]
        row_count, col_count, _ = current_image.shape
        for i in range(salt_pepper_num):
            row = np.random.randint(0, row_count)
            col = np.random.randint(0, col_count)
            current_image[row, col, :] = salt_value
            row = np.random.randint(0, row_count)
            col = np.random.randint(0, col_count)
            current_image[row, col, :] = pepper_value
        aImages[img] = current_image
        if (img % 1000 == 0):
            gc.collect()
            if verbose:
                print(img, ' salted.')
    gc.collect()


class img_folder_dataset:
    def __init__(self, pbasefolder, test_size=0.06, Verbose=False, sizex=256, sizey=256,
                 max_samples_per_class=0, folder_starts_with=''):
        self.basefolder = pbasefolder
        self.verbose = Verbose
        self.test_size = test_size
        self.sizex = sizex
        self.sizey = sizey
        self.max_samples_per_class = max_samples_per_class
        self.folder_starts_with = folder_starts_with

    def load_data(self):
        image_list, label_list = [], []
        folder_list = os.listdir(f"{self.basefolder}/")
        folder_list.sort()
        label_id = 0
        for folder in folder_list:
            if (len(self.folder_starts_with) == 0 or folder.startswith(self.folder_starts_with)):
                cnt_per_class = 0
                img_folder_name = f"{self.basefolder}/{folder}/"
                if self.verbose: print('Loading ' + img_folder_name)
                img_list = os.listdir(img_folder_name)
                for img_file in img_list:
                    absolute_file_name = img_folder_name + '/' + img_file;
                    # print('File:'+absolute_file_name)
                    if absolute_file_name.lower().endswith('.jpg'):
                        aimage = img_to_array(
                            cv2.resize(cv2.imread(absolute_file_name), tuple((self.sizex, self.sizey))), dtype='int16')
                        # print(aimage.shape)
                        image_list.append(aimage)
                        label_list.append(label_id)
                        cnt_per_class = cnt_per_class + 1
                        if (self.max_samples_per_class > 0 and cnt_per_class >= self.max_samples_per_class): break
                label_id = label_id + 1
        image_list = np.array(image_list, dtype='int16')
        label_list = np.array(label_list, dtype='int16')
        x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=self.test_size,
                                                            random_state=17)
        return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def Create2DGaussian(SizeX, SizeY, DistScale=1, ResultScale=1, ResultBias=0):
    rv = scipy.stats.norm()
    s = np.empty((SizeX, SizeY), dtype='float32')
    for i in range(SizeX):
        for j in range(SizeY):
            distance = math.sqrt((i - (SizeX / 2)) ** 2 + (j - (SizeY / 2)) ** 2) * DistScale
            s[i][j] = rv.pdf(distance) * ResultScale + ResultBias
    return s


def Create2DGaussianWithChannels(SizeX, SizeY, DistScale=1, ResultScale=1, ResultBias=0, Channels=1):
    rv = scipy.stats.norm()
    s = np.empty((SizeX, SizeY, Channels), dtype='float32')
    for i in range(SizeX):
        for j in range(SizeY):
            distance = math.sqrt((i - (SizeX / 2)) ** 2 + (j - (SizeY / 2)) ** 2) * DistScale
            cellValue = rv.pdf(distance) * ResultScale + ResultBias
            for k in range(Channels):
                s[i][j][k] = cellValue
    return s


def print_cifar10_result(result):
    labels = np.array([
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'])
    for n in range(9):
        print("[{}] : {}%".format(labels[n], round(result[0][n] * 100, 2)))


def load_dataset(dataset, lab=False, verbose=False, bipolar=True, base_model_name=''):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    if (verbose):
        print("train shape", x_train.shape)
        print("test shape", x_test.shape)
    class_cnt = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, class_cnt)
    y_test = keras.utils.to_categorical(y_test, class_cnt)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if (verbose):
        # Color Images?
        if (len(x_train.shape) == 4):
            for channel in range(0, x_train.shape[3]):
                sub_matrix = x_train[:, :, :, channel]
                print('Original channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
    if (lab):
        if (verbose):
            print("Converting RGB to LAB.")
        # LAB datasets are cached
        cachefilename = 'cache-lab-' + base_model_name + '-' + str(x_train.shape[1]) + '-' + str(
            x_train.shape[2]) + '.npz'
        if not os.path.isfile(cachefilename):
            x_train /= 255
            x_test /= 255
            skimage_rgb2lab_a(x_train, verbose)
            skimage_rgb2lab_a(x_test, verbose)
            gc.collect()
            if (bipolar):
                # JP prefers bipolar input [-2,+2]
                x_train[:, :, :, 0:3] /= [25, 50, 50]
                x_train[:, :, :, 0] -= 2
                x_test[:, :, :, 0:3] /= [25, 50, 50]
                x_test[:, :, :, 0] -= 2
            else:
                x_train[:, :, :, 0:3] /= [100, 200, 200]
                x_train[:, :, :, 1:3] += 0.5
                x_test[:, :, :, 0:3] /= [100, 200, 200]
                x_test[:, :, :, 1:3] += 0.5
            # np.savez_compressed(cachefilename, a=x_train,  b=x_test)
        else:
            loaded = np.load(cachefilename)
            x_train = loaded['a']
            x_test = loaded['b']
        gc.collect()
    else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            x_train /= 64
            x_test /= 64
            x_train -= 2
            x_test -= 2
        else:
            x_train /= 255
            x_test /= 255
    if dataset is fashion_mnist or dataset is mnist:
        img_rows, img_cols = 28, 28
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    gc.collect()
    if (verbose):
        # Color Images?
        if (len(x_train.shape) == 4):
            for channel in range(0, x_train.shape[3]):
                sub_matrix = x_train[:, :, :, channel]
                print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
    return x_train, y_train, x_test, y_test


def load_dataset_with_validation(dataset, lab=False, verbose=False, bipolar=True, base_model_name='',
                                 validation_size=0.1, validation_flip_horizontal=False, validation_flip_vertical=False):
    x_train_full, y_train_full, x_test, y_test = load_dataset(dataset, lab=lab, verbose=verbose, bipolar=bipolar,
                                                              base_model_name=base_model_name)
    x_train_full_size_int = x_train_full.shape[0]
    val_size_int = int(x_train_full_size_int * validation_size)
    # Color Images?
    if (len(x_train_full.shape) == 4):
        x_val = x_train_full[0:val_size_int, :, :, :]
        y_val = y_train_full[0:val_size_int, :]
        x_train = x_train_full[val_size_int:, :, :, :]
        y_train = y_train_full[val_size_int:, :]
    else:
        x_val = x_train_full[0:val_size_int, :, :]
        y_val = y_train_full[0:val_size_int, :]
        x_train = x_train_full[val_size_int:, :, :]
        y_train = y_train_full[val_size_int:, :]

    if (validation_flip_horizontal):
        x_val = np.concatenate((x_val, np.flip(x_val, 2)), axis=0)
        y_val = np.concatenate((y_val, y_val), axis=0)

    if (validation_flip_vertical):
        x_val = np.concatenate((x_val, np.flip(x_val, 1)), axis=0)
        y_val = np.concatenate((y_val, y_val), axis=0)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10_dataset(lab=False, verbose=False, bipolar=True):

    return load_dataset(cifar10, lab=lab, verbose=verbose, bipolar=bipolar, base_model_name='cifar10')


def save_dataset_in_format(aImages, aClasses, dest_folder_name='img', format='.png', with_horizontal_flip=False,
                           with_vertical_flip=False):
    if not os.path.isdir(dest_folder_name):
        os.mkdir(dest_folder_name)
    imgLen = len(aImages)
    for img_cnt in range(imgLen):
        img = aImages[img_cnt]
        class_idx = aClasses[img_cnt].tolist().index(1)
        class_folder = dest_folder_name + '/class_' + str(class_idx)
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)
        cv2.imwrite(class_folder + '/img_' + str(img_cnt) + format, img)
        if (with_horizontal_flip): cv2.imwrite(class_folder + '/h_img_' + str(img_cnt) + format, np.flip(img, 1))
        if (with_vertical_flip): cv2.imwrite(class_folder + '/v_img_' + str(img_cnt) + format, np.flip(img, 0))
        if (with_horizontal_flip and with_vertical_flip): cv2.imwrite(class_folder + '/hv_img_' + str(img_cnt) + format,
                                                                      np.flip(np.flip(img, 0), 1))


def bgr_to_rgb_a(aImages):

    return fix_bad_tfkeras_channel_order(aImages)


def bgr_to_rgb(aImage):

    return fix_img_bad_tfkeras_channel_order(aImage)


def fix_bad_tfkeras_channel_order(aImages):

    local_x = np.zeros(shape=(aImages.shape[0], aImages.shape[1], aImages.shape[2], aImages.shape[3]))
    local_x[:, :, :, 0] = aImages[:, :, :, 2]
    local_x[:, :, :, 1] = aImages[:, :, :, 1]
    local_x[:, :, :, 2] = aImages[:, :, :, 0]
    return local_x


def fix_img_bad_tfkeras_channel_order(aImage):

    local_x = np.zeros(shape=(aImage.shape[0], aImage.shape[1], aImage.shape[2]))
    local_x[:, :, 0] = aImage[:, :, 2]
    local_x[:, :, 1] = aImage[:, :, 1]
    local_x[:, :, 2] = aImage[:, :, 0]
    return local_x


def save_tfds_in_format(p_tfds, dest_folder_name='img', format='.png', with_horizontal_flip=False,
                        with_vertical_flip=False):

    if not os.path.isdir(dest_folder_name):
        os.mkdir(dest_folder_name)
    cnt = 0;
    for x in p_tfds.as_numpy_iterator():
        # print(x["id"])
        # print(x["image"].shape)
        # print(x["label"])
        class_idx = str(x["label"])
        sample_idx = str(cnt)
        img = fix_img_bad_tfkeras_channel_order(x["image"])
        class_folder = dest_folder_name + '/class_' + class_idx
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)
        cv2.imwrite(class_folder + '/img_' + sample_idx + format, img)
        if (with_horizontal_flip): cv2.imwrite(class_folder + '/h_img_' + sample_idx + format, np.flip(img, 1))
        if (with_vertical_flip): cv2.imwrite(class_folder + '/v_img_' + sample_idx + format, np.flip(img, 0))
        if (with_horizontal_flip and with_vertical_flip): cv2.imwrite(class_folder + '/hv_img_' + sample_idx + format,
                                                                      np.flip(np.flip(img, 0), 1))
        cnt = cnt + 1


def save_dataset_as_png(aImages, aClasses, dest_folder_name='img', with_horizontal_flip=False,
                        with_vertical_flip=False):
    save_dataset_in_format(aImages, aClasses, dest_folder_name=dest_folder_name, format='.png',
                           with_horizontal_flip=with_horizontal_flip, with_vertical_flip=with_vertical_flip)


def save_dataset_as_jpg(aImages, aClasses, dest_folder_name='img', with_horizontal_flip=False,
                        with_vertical_flip=False):
    save_dataset_in_format(aImages, aClasses, dest_folder_name=dest_folder_name, format='.jpg',
                           with_horizontal_flip=with_horizontal_flip, with_vertical_flip=with_vertical_flip)


def save_tfds_as_png(p_tfds, dest_folder_name='img', with_horizontal_flip=False, with_vertical_flip=False):
    save_tfds_in_format(p_tfds, dest_folder_name=dest_folder_name, format='.png',
                        with_horizontal_flip=with_horizontal_flip, with_vertical_flip=with_vertical_flip)


def save_tfds_as_jpg(p_tfds, dest_folder_name='img', with_horizontal_flip=False, with_vertical_flip=False):
    save_tfds_in_format(p_tfds, dest_folder_name=dest_folder_name, format='.jpg',
                        with_horizontal_flip=with_horizontal_flip, with_vertical_flip=with_vertical_flip)


def download_file(remote_url, local_file):
    r = requests.get(remote_url, stream=True)
    with open(local_file, "wb") as local_wb:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            # writing one chunk at a time to local file
            if chunk:
                local_wb.write(chunk)


def unzip_file(local_zip_file, expected_folder_name):
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(expected_folder_name)


def download_zip_and_extract(url_zip_file, local_zip_file, expected_folder_name, Verbose=True):
    if not os.path.isfile(local_zip_file):
        if Verbose: print('Downloading: ', url_zip_file, ' to ', local_zip_file)
        download_file(url_zip_file, local_zip_file)
    if not os.path.isdir(expected_folder_name):
        os.mkdir(expected_folder_name)
        if Verbose: print('Decompressing into: ', expected_folder_name)
        unzip_file(local_zip_file, expected_folder_name)


def load_hyperspectral_matlab_image(ImgUrl, ClassUrl, ImgProperty, ClassProperty, LocalBaseName, Verbose=True):
    imgfile = LocalBaseName + '.mat'
    classfile = LocalBaseName + '-class.mat'
    if not os.path.isfile(imgfile):
        if Verbose: print('Downloading:', ImgUrl, ' to ', imgfile)
        urllib.request.urlretrieve(ImgUrl, imgfile)

    if not os.path.isfile(classfile):
        if Verbose: print('Downloading:', ClassUrl, ' to ', classfile)
        urllib.request.urlretrieve(ClassUrl, classfile)

    mat = scipy.io.loadmat(imgfile)
    Image = mat[ImgProperty]
    if Verbose: print('matlab file image shape:', Image.shape)

    mat = scipy.io.loadmat(classfile)
    Classes = mat[ClassProperty]
    if Verbose: print('matlab file classes shape:', Classes.shape)

    Image = Image / Image.max()
    NumClasses = (Classes.max() - Classes.min() + 1)

    return Image, Classes, NumClasses


def slice_image(Image, PixelClasses, NewImageSize=5):
    MaxX = Image.shape[0] - NewImageSize
    MaxY = Image.shape[1] - NewImageSize
    Center = int(NewImageSize / 2)
    aResultImg = [Image[X:X + NewImageSize, Y:Y + NewImageSize, :]
                  for X in range(0, MaxX)
                  for Y in range(0, MaxY)]

    aResultClasses = [int(PixelClasses[X + Center, Y + Center])
                      for X in range(0, MaxX)
                      for Y in range(0, MaxY)]

    NumClasses = (PixelClasses.max() - PixelClasses.min() + 1)
    aResultClasses = keras.utils.to_categorical(aResultClasses, NumClasses)

    return np.array(aResultImg), aResultClasses


def slice_images(Images, NewImageSize=3):
    ImageCount = Images.shape[0]
    MaxX = Images.shape[1] - NewImageSize
    MaxY = Images.shape[2] - NewImageSize
    aResultImg = [Images[ImagePos, X:X + NewImageSize, Y:Y + NewImageSize, :]
                  for ImagePos in range(0, ImageCount)
                  for X in range(0, MaxX)
                  for Y in range(0, MaxY)]
    return np.array(aResultImg)


def create_pixel_array_from_3D_image(Image):
    SizeX = Image.shape[0]
    SizeY = Image.shape[1]
    Channels = Image.shape[2]
    aResultImage = Image.reshape(SizeX * SizeY, Channels)
    return aResultImage


def create_pixel_array_from_3D_images(Images):
    ImageCount = Images.shape[0]
    SizeX = Images.shape[1]
    SizeY = Images.shape[2]
    Channels = Images.shape[3]
    aResult = Images.reshape(ImageCount * SizeX * SizeY, Channels)
    return np.array(aResult)


def create_3D_image_from_pixel_array(PixelArray, SizeX, SizeY, Depth=3):
    aResultImage = PixelArray.reshape(SizeX, SizeY, Depth)
    return aResultImage


def create_image_generator_sliced_image():
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.,
        zoom_range=[1.0, 1.0],
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)
    return datagen


def train_model_on_dataset(model, dataset, base_model_name, plrscheduler, batch_size=64,
                           epochs=300, momentum=0.9, nesterov=True, verbose=False, lab=False, bipolar=True,
                           datagen=utils.create_image_generator(), monitor='val_accuracy',
                           use_multiprocessing=False):
    x_train, y_train, x_test, y_test = data.datasets.load_dataset(dataset, verbose=verbose, lab=lab, bipolar=bipolar,
                                                                 base_model_name=base_model_name)
    gc.collect()

    batches_per_epoch = np.floor(x_train.shape[0] / batch_size)
    batches_per_validation = np.floor(x_test.shape[0] / batch_size)
    model_name = base_model_name + '.h5'
    csv_name = base_model_name + '.csv'
    opt = keras.optimizers.SGD(lr=0.1, momentum=momentum, nesterov=nesterov)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    fit_verbose = 0
    if (verbose):
        fit_verbose = 2

    gc.collect()

    fit_result = model.fit(
        x=datagen.flow(x_train, y_train, batch_size=batch_size),
        batch_size=batch_size,
        epochs=epochs,
        verbose=fit_verbose,
        callbacks=[
            keras.callbacks.LearningRateScheduler(plrscheduler),
            keras.callbacks.ModelCheckpoint(
                filepath=model_name,
                monitor=monitor,
                verbose=fit_verbose,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                save_freq='epoch')
        ],
        steps_per_epoch=batches_per_epoch,
        validation_steps=batches_per_validation,
        validation_data=(x_test, y_test),
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=use_multiprocessing
    )
    gc.collect()
    return fit_result, model_name, csv_name


def train_model_on_cifar10(model, base_model_name, plrscheduler, batch_size=64, epochs=300, momentum=0.9, nesterov=True,
                           verbose=False, lab=False, bipolar=True):
    return train_model_on_dataset(model=model, dataset=cifar10, base_model_name=base_model_name,
                                  plrscheduler=plrscheduler, batch_size=batch_size, epochs=epochs, momentum=momentum,
                                  nesterov=nesterov,
                                  verbose=verbose, lab=lab, bipolar=bipolar)


def load_image_file_names_from_folder(img_folder_name):
    img_list = os.listdir(img_folder_name)
    output_img_list = []
    for img_file in img_list:
        absolute_file_name = img_folder_name + '/' + img_file
        absolute_file_name_lower = absolute_file_name.lower()
        if absolute_file_name_lower.endswith('.jpg') or absolute_file_name_lower.endswith(
                '.jepg') or absolute_file_name_lower.endswith('.png'):
            output_img_list.append(absolute_file_name)
    return output_img_list


def flip_images_on_folder(img_folder_name, with_horizontal_flip=True, with_vertical_flip=False):
    img_list = os.listdir(img_folder_name)
    for img_file in img_list:
        absolute_file_name = img_folder_name + '/' + img_file
        absolute_file_name_lower = absolute_file_name.lower()
        if absolute_file_name_lower.endswith('.jpg') or absolute_file_name_lower.endswith(
                '.jepg') or absolute_file_name_lower.endswith('.png'):
            img = load_img(absolute_file_name)
            img = img_to_array(img, dtype='float32')
            h_absolute_dest_file_name = img_folder_name + '/h_' + img_file
            v_absolute_dest_file_name = img_folder_name + '/v_' + img_file
            hv_absolute_dest_file_name = img_folder_name + '/hv_' + img_file
            if (with_horizontal_flip): cv2.imwrite(h_absolute_dest_file_name, np.flip(img, 1))
            if (with_vertical_flip): cv2.imwrite(v_absolute_dest_file_name, np.flip(img, 0))
            if (with_horizontal_flip and with_vertical_flip): cv2.imwrite(hv_absolute_dest_file_name,
                                                                          np.flip(np.flip(img, 0), 1))


def add_padding_to_make_img_array_squared(img):
    sizex = img.shape[0]
    sizey = img.shape[1]
    if (sizex == sizey):
        return img
    else:
        maxsize = np.max([sizex, sizey])
        padx = (maxsize - sizex) // 2
        pady = (maxsize - sizey) // 2
        return np.pad(img, pad_width=((padx, padx), (pady, pady), (0, 0)))


def load_images_from_files(file_names, target_size=(224, 224), dtype='float32', smart_resize=False, lab=False,
                           rescale=False, bipolar=False):

    def local_rescale(img, lab):
        if (lab):
            if (bipolar):
                img[:, :, 0:3] /= [25, 50, 50]
                img[:, :, 0] -= 2
            else:
                img[:, :, 0:3] /= [100, 200, 200]
                img[:, :, 1:3] += 0.5
        else:
            if (bipolar):
                img /= 64
                img -= 2
            else:
                img /= 255

    x = []
    cnt = 0
    for file_name in file_names:
        cnt = cnt + 1
        if (cnt % 1000 == 0):
            gc.collect()
        if (smart_resize):
            img = load_img(file_name)
            img = img_to_array(img, dtype='float32')
            if (lab):
                img /= 255
                img = skimage_color.rgb2lab(img)
            if (rescale):
                local_rescale(img, lab)
            img = add_padding_to_make_img_array_squared(img)
            if ((img.shape[0] != target_size[0]) or (img.shape[1] != target_size[1])):
                img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        else:
            img = load_img(file_name, target_size=target_size)
            img = img_to_array(img, dtype='float32')
            if (lab):
                img /= 255
                img = skimage_color.rgb2lab(img)
            if (rescale):
                local_rescale(img, lab)
        x.append(img)
    return np.array(x, dtype=dtype)


def load_images_from_folders(seed=None, root_dir=None, lab=False,
                             verbose=True, bipolar=False, base_model_name='plant_leaf',
                             training_size=0.6, validation_size=0.2, test_size=0.2,
                             target_size=(224, 224),
                             has_training=True, has_validation=True, has_testing=True,
                             smart_resize=False):
    if root_dir is None:
        print("No root dir at load_images_from_folders")
        return

    if seed is not None:
        random.seed(seed)

    classes = os.listdir(root_dir)
    classes = sorted(classes)

    classes_num = len(classes)
    if (verbose):
        print("Loading ", classes_num, " classes.")

    train_path = []
    val_path = []
    test_path = []

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for i, _class in enumerate(classes):
        paths = glob.glob(os.path.join(root_dir, _class, "*"))
        paths = [n for n in paths if
                 n.lower().endswith(".png") or
                 n.lower().endswith(".jpg") or
                 n.lower().endswith(".jpeg") or
                 n.lower().endswith(".tif") or
                 n.lower().endswith(".tiff")
                 ]
        random.shuffle(paths)
        cat_total = len(paths)
        if (training_size > 0):
            train_path.extend(paths[:int(cat_total * training_size)])
            train_y.extend([i] * int(cat_total * training_size))
        if (validation_size > 0):
            val_path.extend(paths[int(cat_total * training_size):int(cat_total * (training_size + validation_size))])
            val_y.extend(
                [i] * len(paths[int(cat_total * training_size):int(cat_total * (training_size + validation_size))]))
        if (test_size > 0):
            if (training_size + validation_size + test_size >= 1):
                test_path.extend(paths[int(cat_total * (training_size + validation_size)):])
                test_y.extend([i] * len(paths[int(cat_total * (training_size + validation_size)):]))
            else:
                test_path.extend(paths[int(cat_total * (training_size + validation_size)):int(
                    cat_total * (training_size + validation_size + test_size))])
                test_y.extend([i] * len(paths[int(cat_total * (training_size + validation_size)):int(
                    cat_total * (training_size + validation_size + test_size))]))

    if (verbose and smart_resize):
        print("smart resize is enabled.")

    if has_training:
        if (verbose):
            print("loading train images")
        train_x = np.array(
            data.datasets.load_images_from_files(train_path, target_size=target_size, smart_resize=smart_resize, lab=lab,
                                                rescale=True, bipolar=bipolar), dtype='float32')
        if (verbose):
            print("train shape is:", train_x.shape)
    else:
        train_x = np.array([])

    if has_validation:
        if (verbose):
            print("loading validation images")
        val_x = np.array(
            data.datasets.load_images_from_files(val_path, target_size=target_size, smart_resize=smart_resize, lab=lab,
                                                rescale=True, bipolar=bipolar), dtype='float32')
        if (verbose):
            print("validation shape is:", val_x.shape)
    else:
        val_x = np.array([])

    if has_testing:
        if (verbose):
            print("loading test images")
        test_x = np.array(
            data.datasets.load_images_from_files(test_path, target_size=target_size, smart_resize=smart_resize, lab=lab,
                                                rescale=True, bipolar=bipolar), dtype='float32')
        if (verbose):
            print("test shape is:", test_x.shape)
    else:
        test_x = np.array([])

    train_y = np.array(train_y)
    val_y = np.array(val_y)
    test_y = np.array(test_y)

    if (verbose and has_training):
        for channel in range(0, train_x.shape[3]):
            sub_matrix = train_x[:, :, :, channel]
            print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))

    class_weight = None
    if has_training:
        local_class_weight = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_y),
            y=train_y)
        class_weight = {i: local_class_weight[i] for i in range(classes_num)}
    train_y = keras.utils.to_categorical(train_y, classes_num)
    val_y = keras.utils.to_categorical(val_y, classes_num)
    test_y = keras.utils.to_categorical(test_y, classes_num)
    if (verbose):
        print("Loaded.")

    return train_x, val_x, test_x, train_y, val_y, test_y, class_weight, classes


def print_classification_report(pred_y, test_y):
    pred_classes_y = np.array(list(np.argmax(pred_y, axis=1)))
    test_classes_y = np.array(list(np.argmax(test_y, axis=1)))
    report = classification_report(test_classes_y, pred_classes_y, digits=4)
    print(report)


def rgb_to_black_white_a(test_x):
    bw_test = np.copy(test_x)
    bw_test[:, :, :, 0] += test_x[:, :, :, 1] + test_x[:, :, :, 2]
    bw_test[:, :, :, 0] /= 3
    bw_test[:, :, :, 1] = bw_test[:, :, :, 0]
    bw_test[:, :, :, 2] = bw_test[:, :, :, 0]
    return bw_test


def test_flips_on_model(test_x, test_y, model, has_flip_x=True, has_flip_y=True, has_bw=True, center_crop=0.0):
    print("Test Original")
    pred_y = model.predict(test_x)
    print_classification_report(pred_y, test_y)

    if (has_flip_x):
        print("Test Flip X")
        test_x_flip_x = np.flip(test_x, 2)
        pred_y_fx = model.predict(test_x_flip_x)
        print_classification_report(pred_y_fx, test_y)

        print("Test Original + Flip X")
        print_classification_report(pred_y + pred_y_fx, test_y)

    if (has_flip_y):
        print("Test Flip Y")
        test_x_flip_y = np.flip(test_x, 1)
        pred_y_fy = model.predict(test_x_flip_y)
        print_classification_report(pred_y_fy, test_y)

        print("Test Original + Flip Y")
        print_classification_report(pred_y + pred_y_fy, test_y)

    if (has_flip_x and has_flip_y):
        print("Test Original + Flip X + Flip Y")
        print_classification_report(pred_y + pred_y_fx + pred_y_fy, test_y)

    if (center_crop > 0):
        print("Cropped and Resized")
        size_x = test_x.shape[2]
        size_y = test_x.shape[1]
        center_crop_px = int((size_x * (center_crop)) / 2)
        center_crop_py = int((size_y * (center_crop)) / 2)
        cropped_test_x = test_x[:, center_crop_py: size_y - center_crop_py, center_crop_px:size_x - center_crop_px, :]
        print("Cropped shape:", cropped_test_x.shape)
        crop_resized = cv2_resize_a(cropped_test_x, target_size=(size_y, size_x))
        pred_y_cr = model.predict(crop_resized)
        print_classification_report(pred_y_cr, test_y)
        print("Original + Cropped Resized")
        print_classification_report(pred_y + pred_y_cr, test_y)
        if (has_flip_x):
            print("Original + Flip X + Cropped Resized")
            print_classification_report(pred_y + pred_y_fx + pred_y_cr, test_y)

    if (has_bw):
        print("Test Black and White")
        bw_test = rgb_to_black_white_a(test_x)
        pred_y_bw = model.predict(bw_test)
        print_classification_report(pred_y_bw, test_y)

        print("Test Original + Black and White")
        print_classification_report(pred_y + pred_y_bw, test_y)

    if (has_flip_x and has_flip_y and has_bw):
        print("Test Original + Flip X + Flip Y + BW")
        print_classification_report(pred_y + pred_y_fx + pred_y_fy + pred_y_bw, test_y)


def test_flips_on_saved_model(test_x, test_y, model_file_name, has_flip_x=True, has_flip_y=True, has_bw=True,
                              center_crop=0.0):
    models = model.models.load_kereas_model(model_file_name)
    test_flips_on_model(test_x=test_x, test_y=test_y, model=models, has_flip_x=has_flip_x, has_flip_y=has_flip_y,
                        has_bw=has_bw, center_crop=center_crop)


def clone_sub_folders(source_folder, dest_folder, verbose=False):
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
        if verbose: print("Creating folder " + dest_folder)
    subfolders = os.listdir(source_folder)
    for subfolder in subfolders:
        if os.path.isdir(source_folder + '/' + subfolder):
            new_folder = dest_folder + '/' + subfolder
            if not os.path.isdir(new_folder):
                if verbose: print("Creating folder " + new_folder)
                os.mkdir(new_folder)


def extract_subset_every(source_folder, dest_folder, move_every=10, shift=0, verbose=False):
    clone_sub_folders(source_folder, dest_folder, verbose)
    subfolders = os.listdir(source_folder)
    for subfolder in subfolders:
        class_folder = source_folder + '/' + subfolder
        dest_class_folder = dest_folder + '/' + subfolder
        file_list = sorted(
            [filename for filename in os.listdir(class_folder) if os.path.isfile(class_folder + '/' + filename)])
        class_file_cnt = len(file_list)
        start_pos = shift
        file_pos = start_pos
        moved_cnt = 0
        while file_pos < class_file_cnt:
            source_file = class_folder + '/' + file_list[file_pos]
            dest_file = dest_class_folder + '/' + file_list[file_pos]
            if not os.path.isfile(dest_file):
                shutil.move(source_file, dest_file)
                moved_cnt += 1
            file_pos += move_every
        if verbose: print(
            str(moved_cnt) + " files have been moved from " + class_folder + " to " + dest_class_folder + ".")
