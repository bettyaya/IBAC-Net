import numpy as np
import os
import tensorflow
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import color as skimage_color
import csv
import random


def save_2d_array_as_csv(a, filename):
    with open(filename, "w+") as local_csv:
        csvWriter = csv.writer(local_csv, delimiter=',')
        csvWriter.writerows(a)


def slice_3d_into_2d(aImage, NumRows, NumCols, ForceCellMax=False):
    SizeX = aImage.shape[0]
    SizeY = aImage.shape[1]
    Depth = aImage.shape[2]
    NewSizeX = SizeX * NumCols
    NewSizeY = SizeY * NumRows
    aResult = np.zeros(shape=(NewSizeX, NewSizeY))
    # print(aResult.shape)
    for depthCnt in range(Depth):
        PosX = depthCnt % NumCols
        PosY = int(depthCnt / NumCols)
        # print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
        if ForceCellMax:
            Slice = aImage[:, :, depthCnt]
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY] += Slice
        else:
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY] += aImage[:, :, depthCnt]
    return aResult


def slice_3d_into_2d_cl(aImage, NumRows, NumCols, ForceCellMax=False):
    SizeX = aImage.shape[1]
    SizeY = aImage.shape[2]
    Depth = aImage.shape[0]
    NewSizeX = SizeX * NumCols
    NewSizeY = SizeY * NumRows
    aResult = np.zeros(shape=(NewSizeX, NewSizeY))
    for depthCnt in range(Depth):
        PosX = depthCnt % NumCols
        PosY = int(depthCnt / NumCols)
        if ForceCellMax:
            Slice = aImage[depthCnt, :, :]
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY] += Slice
        else:
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY] += aImage[depthCnt, :, :]
    return aResult


def slice_4d_into_3d(aImage, NumRows, NumCols, ForceCellMax=False):
    SizeX = aImage.shape[0]
    SizeY = aImage.shape[1]
    Depth = aImage.shape[2]
    Neurons = aImage.shape[3]
    NewSizeX = SizeX * NumCols
    NewSizeY = SizeY * NumRows
    aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
    for NeuronsCnt in range(Neurons):
        PosX = NeuronsCnt % NumCols
        PosY = int(NeuronsCnt / NumCols)
        if ForceCellMax:
            Slice = aImage[:, :, :, NeuronsCnt]
            Slice = Slice - Slice.min()
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY, :] += Slice
        else:
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY, :] += aImage[:, :, :, NeuronsCnt]
    return aResult


def slice_4d_into_3d_cl(aImage, NumRows, NumCols, ForceCellMax=False):
    SizeX = aImage.shape[1]
    SizeY = aImage.shape[2]
    Depth = aImage.shape[3]
    Neurons = aImage.shape[0]
    NewSizeX = SizeX * NumCols
    NewSizeY = SizeY * NumRows
    aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
    for NeuronsCnt in range(Neurons):
        PosX = NeuronsCnt % NumCols
        PosY = int(NeuronsCnt / NumCols)
        if ForceCellMax:
            Slice = aImage[NeuronsCnt, :, :, :]
            Slice = Slice - Slice.min()
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY, :] += Slice
        else:
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY, :] += aImage[NeuronsCnt, :, :, :]
    return aResult


def show_neuronal_patterns(aWeights, NumRows, NumCols, ForceCellMax=False):
    SizeX = aWeights.shape[0]
    SizeY = aWeights.shape[1]
    Depth = aWeights.shape[2]
    Neurons = aWeights.shape[3]
    NewSizeX = SizeX * NumCols + NumCols - 1
    NewSizeY = SizeY * NumRows + NumRows - 1
    aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
    for NeuronsCnt in range(Neurons):
        PosX = NeuronsCnt % NumCols
        PosY = int(NeuronsCnt / NumCols)
        if ForceCellMax:
            Slice = aWeights[:, :, :, NeuronsCnt]
            Slice = Slice - Slice.min()
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX + PosX * SizeX:PosX + (PosX + 1) * SizeX, PosY + PosY * SizeY:PosY + (PosY + 1) * SizeY,
            :] += Slice
        else:
            aResult[PosX + PosX * SizeX:PosX + (PosX + 1) * SizeX, PosY + PosY * SizeY:PosY + (PosY + 1) * SizeY,
            :] += aWeights[:, :, :, NeuronsCnt]
    return aResult


def show_neuronal_patterns_nf(aWeights, NumRows, NumCols, ForceCellMax=False):
    Neurons = aWeights.shape[0]
    SizeX = aWeights.shape[1]
    SizeY = aWeights.shape[2]
    Depth = aWeights.shape[3]
    NewSizeX = SizeX * NumCols + NumCols - 1
    NewSizeY = SizeY * NumRows + NumRows - 1
    aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
    for NeuronsCnt in range(Neurons):
        PosX = NeuronsCnt % NumCols
        PosY = int(NeuronsCnt / NumCols)
        if ForceCellMax:
            Slice = aWeights[NeuronsCnt, :, :, :]
            Slice = Slice - Slice.min()
            SliceMax = Slice.max()
            if SliceMax > 0:
                Slice /= SliceMax
            aResult[PosX + PosX * SizeX:PosX + (PosX + 1) * SizeX, PosY + PosY * SizeY:PosY + (PosY + 1) * SizeY,
            :] += Slice
        else:
            aResult[PosX + PosX * SizeX:PosX + (PosX + 1) * SizeX, PosY + PosY * SizeY:PosY + (PosY + 1) * SizeY,
            :] += aWeights[NeuronsCnt, :, :, :]
    return aResult


def slice_4d_into_2d(aImage, ForceColMax=False, ForceRowMax=False, ForceCellMax=False):
    Images = aImage.shape[0]
    SizeX = aImage.shape[1]
    SizeY = aImage.shape[2]
    Depth = aImage.shape[3]
    NewSizeX = SizeX * Depth
    NewSizeY = SizeY * Images
    aResult = np.zeros(shape=(NewSizeX, NewSizeY))
    for depthCnt in range(Depth):
        if ForceRowMax:
            SliceMin = 0
            SliceMax = aImage[:, :, :, depthCnt].max() - SliceMin
        for imgCnt in range(Images):
            PosX = depthCnt
            PosY = imgCnt
            Slice = np.copy(aImage[imgCnt, :, :, depthCnt])
            if ForceColMax:
                SliceMin = 0
                SliceMax = aImage[imgCnt, :, :, :].max() - SliceMin

            if ForceCellMax:
                SliceMin = 0
                SliceMax = Slice.max() - SliceMin

            if ForceRowMax or ForceColMax or ForceCellMax:
                Slice -= SliceMin
                if SliceMax > 0:
                    Slice /= SliceMax
            aResult[PosX * SizeX:(PosX + 1) * SizeX, PosY * SizeY:(PosY + 1) * SizeY] += Slice
    return aResult


def evaluate_model_print(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss", scores[0])
    print("Test accuracy", scores[1])
    return scores


def create_folder_if_required(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


def get_model_parameter_counts(model):
    trainable_count = int(np.sum([backend.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([backend.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, non_trainable_count


def preprocess(img, bipolar=True, tfcons=False):
    if (bipolar):
        img /= 64
        img -= 2
    else:
        img /= 255
    if (tfcons): img = tensorflow.constant(np.array(img))


def preprocess_cp(img, bipolar=True, tfcons=False):
    img_result = np.copy(img)
    preprocess(img_result, bipolar=bipolar, tfcons=tfcons)
    return img_result


def deprocess(img, bipolar=True, tfcast=False):
    if (bipolar):
        img += 2
        img *= 64
    else:
        img *= 255
    if (tfcast): img = tensorflow.cast(img, tensorflow.uint8)


def deprocess_cp(img, bipolar=True, tfcast=False):
    img_result = np.copy(img)
    deprocess_cp(img_result, bipolar=bipolar, tfcast=tfcast)
    return img_result


def rgb2monopolar(img):
    img /= 255
    return img


def rgb2bipolar(img):
    img /= 64
    img -= 2
    return img


def rgb2monopolar_lab(img):
    img /= 255
    img = skimage_color.rgb2lab(img)
    img[:, :, 0:3] /= [100, 200, 200]
    img[:, :, 1:3] += 0.5
    return img


def rgb2bipolar_lab(img):
    img /= 255
    img = skimage_color.rgb2lab(img)
    img[:, :, 0:3] /= [25, 50, 50]
    img[:, :, 0] -= 2
    return img


def rgb2black_white_25percent(img):
    if random.randint(0, 100) < 25:
        bw_test = np.copy(img)
        bw_test[:, :, 0] += img[:, :, 1] + img[:, :, 2]
        bw_test[:, :, 0] /= 3
        bw_test[:, :, 1] = bw_test[:, :, 0]
        bw_test[:, :, 2] = bw_test[:, :, 0]
        return bw_test
    else:
        return img


def rgb2black_white_50percent(img):
    if random.randint(0, 100) < 50:
        bw_test = np.copy(img)
        bw_test[:, :, 0] += img[:, :, 1] + img[:, :, 2]
        bw_test[:, :, 0] /= 3
        bw_test[:, :, 1] = bw_test[:, :, 0]
        bw_test[:, :, 2] = bw_test[:, :, 0]
        return bw_test
    else:
        return img


def rgb2black_white_75percent(img):
    if random.randint(0, 100) < 75:
        bw_test = np.copy(img)
        bw_test[:, :, 0] += img[:, :, 1] + img[:, :, 2]
        bw_test[:, :, 0] /= 3
        bw_test[:, :, 1] = bw_test[:, :, 0]
        bw_test[:, :, 2] = bw_test[:, :, 0]
        return bw_test
    else:
        return img


def create_image_generator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.3,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
):
    return ImageDataGenerator(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split)


def create_image_generator_no_augmentation(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=0,  #
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
):

    return create_image_generator(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,  #
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split
    )


def relu(adata):
    return np.maximum(0, adata)


def reverse_sort(arraydata):
    return np.array(list(reversed(np.sort(arraydata))))


def get_class_position(pclass, predictions):
    predicted_probability = predictions[pclass]
    predictions_sorted = reverse_sort(predictions)
    return np.where(predictions_sorted == predicted_probability)[0][0]


def get_max_acceptable_common_divisor(a, b, max_acceptable=1000000):
    divisor = max(1, min(a, b, max_acceptable))
    while divisor > 0:
        if a % divisor == 0 and b % divisor == 0:
            return divisor
            break
        divisor -= 1
