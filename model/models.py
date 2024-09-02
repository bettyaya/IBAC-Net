from keras.src.layers import Dropout
import layers
import utils
import inception_v3
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Flatten, Dense
from trans import TransformerEncoder  # 导入 Transformer 编码器
import model

def residual_block(x, filters, kernel_size=3, strides=1, l2_decay=0.0):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding='same',
               kernel_regularizer=keras.regularizers.l2(l2_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(l2_decay))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def save_model(model, base_name):
    text_file = open(base_name + '.model', "w")
    text_file.write(model.to_json())
    text_file.close()
    model.save_weights(base_name + '.h5')


def load_model(base_name):
    model = Model()
    text_file = open(base_name + '.model', "r")
    config = text_file.read()
    model = model_from_json(config, {'CopyChannels': layers.CopyChannels})
    model.load_weights(base_name + '.h5')
    return model


def load_kereas_model(filename):
    return keras.models.load_model(filename, custom_objects=layers.GetClasses())


def make_model_trainable(model):
    for layer in model.layers:
        layer.trainable = True


def make_model_untrainable(model):
    for layer in model.layers:
        layer.trainable = False


def plant_leaf(pinput_shape, num_classes, l2_decay=0.0, dropout_drop_rate=0.2, has_batch_norm=True):
    img_input = Input(shape=pinput_shape)
    x = Conv2D(32, (3, 3), padding='valid', input_shape=pinput_shape,
               kernel_regularizer=keras.regularizers.l2(l2_decay))(img_input)
    if has_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, strides=2)(x)

    x = residual_block(x, filters=32, kernel_size=3, l2_decay=l2_decay)
    x = residual_block(x, filters=32, kernel_size=3, l2_decay=l2_decay)

    x = Conv2D(16, (3, 3), padding='valid', kernel_regularizer=keras.regularizers.l2(l2_decay))(x)
    if has_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, strides=2)(x)

    x = residual_block(x, filters=16, kernel_size=3, l2_decay=l2_decay)
    x = residual_block(x, filters=16, kernel_size=3, l2_decay=l2_decay)

    x = Conv2D(8, (3, 3), padding='valid', kernel_regularizer=keras.regularizers.l2(l2_decay))(x)
    if has_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, strides=2)(x)

    x = residual_block(x, filters=8, kernel_size=3, l2_decay=l2_decay)
    x = residual_block(x, filters=8, kernel_size=3, l2_decay=l2_decay)

    x = Flatten()(x)

    transformer_layer = TransformerEncoder(
        name='transformer_encoder',
        input_tensor=x,
        num_heads=8,
        dff=64,
        num_transformer_blocks=4,
        mlp_units=[128],
        rate=0.1,
    )

    x = Dense(128)(transformer_layer)
    if dropout_drop_rate > 0.0:
        x = Dropout(rate=dropout_drop_rate)(x)
    x = Activation('relu')(x)

    output = Dense(num_classes, activation='softmax', name='softmax')(x)

    return Model(inputs=img_input, outputs=output)


def CreatePartialModel(pModel, pOutputLayerName, hasGlobalAvg=False):
    inputs = pModel.input
    outputs = pModel.get_layer(pOutputLayerName).output
    if (hasGlobalAvg):
        outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def CreatePartialModelCopyingChannels(pModel, pOutputLayerName, pChannelStart, pChannelCount):
    inputs = pModel.input
    outputs = pModel.get_layer(pOutputLayerName).output
    outputs = layers.CopyChannels(channel_start=pChannelStart, channel_count=pChannelCount)(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def CreatePartialModelWithSoftMax(pModel, pOutputLayerName, numClasses, newLayerName='k_probs'):
    models = model.models.CreatePartialModel(pModel, pOutputLayerName)
    inputs = models.input
    outputs = models.get_layer(pOutputLayerName).output
    outputs = keras.layers.Dense(numClasses,
                                 activation='softmax',
                                 name=newLayerName)(outputs)
    models = keras.models.Model(inputs, outputs)
    return models


def CreatePartialModelFromChannel(pModel, pOutputLayerName, pChannelIdx):

    return CreatePartialModelCopyingChannels(pModel, pOutputLayerName, pChannelStart=pChannelIdx, pChannelCount=1)


def PartialModelPredict(aInput, pModel, pOutputLayerName, hasGlobalAvg=False, pBatchSize=32):
    inputs = pModel.input
    outputs = pModel.get_layer(pOutputLayerName).output
    if (hasGlobalAvg):
        outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    IntermediateLayerModel = keras.Model(inputs=inputs, outputs=outputs)
    layeroutput = np.array(IntermediateLayerModel.predict(x=aInput, batch_size=pBatchSize))
    IntermediateLayerModel = 0
    return layeroutput


def calculate_heat_map_from_dense_and_avgpool(aInput, target_class, pModel, pOutputLayerName, pDenseLayerName):
    localImageArray = []
    localImageArray.append(aInput)
    localImageArray = np.array(localImageArray)
    class_weights = pModel.get_layer(pDenseLayerName).get_weights()[0]
    conv_output = model.models.PartialModelPredict(localImageArray, pModel, pOutputLayerName)[0]
    a_heatmap_result = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    # print(a_heatmap_result.shape)
    # print(type(conv_output[:, :, 0]))
    # print(conv_output[:, :, 0].shape)
    for i, w in enumerate(class_weights[:, target_class]):
        a_heatmap_result += w * conv_output[:, :, i]
    a_heatmap_result = utils.relu(a_heatmap_result)
    max_heatmap_result = np.max(a_heatmap_result)
    if max_heatmap_result > 0:
        a_heatmap_result = a_heatmap_result / max_heatmap_result
    return a_heatmap_result


def compiled_full_two_path_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        max_mix_idx=10,
        model_name='two_path_inception_v3'):

    return inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=True,
        two_paths_second_block=True,
        deep_two_paths=True,
        deep_two_paths_compression=0.655,
        max_mix_idx=max_mix_idx,
        model_name='deep_two_path_inception_v3'
    )


def compiled_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        max_mix_idx=10,
        model_name='two_path_inception_v3'):

    return inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        max_mix_idx=max_mix_idx,
        model_name='two_path_inception_v3'
    )


def compiled_two_path_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        deep_two_paths_compression=0.655,
        deep_two_paths_bottleneck_compression=0.5,
        l_ratio=0.5,
        ab_ratio=0.5,
        max_mix_idx=10,
        max_mix_deep_two_paths_idx=-1,
        model_name='two_path_inception_v3'
):

    return inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=two_paths_partial_first_block,
        two_paths_first_block=two_paths_first_block,
        two_paths_second_block=two_paths_second_block,
        deep_two_paths=deep_two_paths,
        deep_two_paths_compression=deep_two_paths_compression,
        deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
        l_ratio=l_ratio,
        ab_ratio=ab_ratio,
        max_mix_idx=max_mix_idx,
        max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
        model_name=model_name)


def create_paths(last_tensor, compression, l2_decay, dropout_rate=0.0):
    bn_axis = 3
    last_tensor = keras.layers.Conv2D(int(keras.backend.int_shape(last_tensor)[bn_axis] * compression), 1,
                                      use_bias=True, activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    if (dropout_rate > 0): last_tensor = keras.layers.Dropout(dropout_rate)(last_tensor)
    last_tensor = keras.layers.MaxPooling2D(2, strides=2)(last_tensor)
    return last_tensor


def two_path_inception_v3(
        include_top=True,
        weights=None,  # 'two_paths_plant_leafs'
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        deep_two_paths_compression=0.655,
        deep_two_paths_bottleneck_compression=0.5,
        l_ratio=0.5,
        ab_ratio=0.5,
        max_mix_idx=10,
        max_mix_deep_two_paths_idx=-1,
        model_name='two_path_inception_v3',
        **kwargs):

    return inception_v3.two_path_inception_v3(
        include_top=include_top,
        weights=weights,  # 'two_paths_plant_leafs'
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        two_paths_partial_first_block=two_paths_partial_first_block,
        two_paths_first_block=two_paths_first_block,
        two_paths_second_block=two_paths_second_block,
        deep_two_paths=deep_two_paths,
        deep_two_paths_compression=deep_two_paths_compression,
        deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
        l_ratio=l_ratio,
        ab_ratio=ab_ratio,
        max_mix_idx=max_mix_idx,
        max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
        model_name=model_name)
