import skimage
import tensorflow as tf
import sys
import data.datasets
import layers
import model.models
import tensorflow
import multiprocessing
import shutil
import os
from sklearn.metrics import f1_score

monitor = 'val_accuracy'
epochs = 30
batch_size = 16
input_shape = (128, 128, 3)
Verbose = True
import time
start_time = time.time()
import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)
data_dir = ""
print(os.listdir(data_dir))

train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = data.datasets.load_images_from_folders(seed=7, root_dir=data_dir, lab=True,
  verbose=Verbose, bipolar=False, base_model_name='plant_leaf',
  training_size=0.6, validation_size=0.2, test_size=0.2,
  target_size=(input_shape[0],input_shape[1]),
  has_training=True, has_validation=True, has_testing=True,
  smart_resize=True)

print(train_x.shape,val_x.shape,test_x.shape)
print(train_y.shape,val_y.shape,test_y.shape)

for two_paths_second_block in [False]:
  for l_ratio in [0.2]:
    basefilename = 'two-path-inception-v2.8-'+str(two_paths_second_block)+'-'+str(l_ratio)
    print('Running: '+basefilename)
    model = model.models.compiled_two_path_inception_v3(
      input_shape=input_shape,
      classes=34,
      two_paths_first_block=True,
      two_paths_second_block=two_paths_second_block,
      l_ratio=l_ratio,
      ab_ratio=(1-l_ratio),
      max_mix_idx=5,
      model_name='two_path_inception_v3'
      )
    monitor='val_accuracy'
    best_result_file_name = basefilename+'-best_result.hdf5'
    save_best = tensorflow.keras.callbacks.ModelCheckpoint(
      filepath=best_result_file_name,
      monitor=monitor,
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='max',
      period=1)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
      validation_data=(val_x,val_y),
      callbacks=[save_best],class_weight=classweight,
      workers=multiprocessing.cpu_count())
    val_loss, val_accuracy = model.evaluate(val_x, val_y, verbose=0)
    val_y_pred = model.predict(val_x)
    val_y_pred_labels = tensorflow.argmax(val_y_pred, axis=-1)
    val_y_labels = tensorflow.argmax(val_y, axis=-1)
    f1_val = f1_score(val_y_labels, val_y_pred_labels, average='weighted')
    print(
      f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {f1_val:.4f}')
    print('Testing Last Model: '+basefilename)
    evaluated = model.evaluate(test_x,test_y)
    for metric, name in zip(evaluated,["loss","acc","top 5 acc"]):
      print(name,metric)
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
    test_y_pred = model.predict(test_x)
    test_y_pred_labels = tensorflow.argmax(test_y_pred, axis=-1)
    test_y_labels = tensorflow.argmax(test_y, axis=-1)
    f1_test = f1_score(test_y_labels, test_y_pred_labels, average='weighted')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {f1_test:.4f}')
    print('Best Model Results: '+basefilename)
    model = tensorflow.keras.models.load_model(best_result_file_name, custom_objects={'CopyChannels': layers.CopyChannels})
    evaluated = model.evaluate(test_x,test_y)
    model.models.save_model(model, basefilename)
    for metric, name in zip(evaluated,["loss","acc","top 5 acc"]):
      print(name,metric)
    val_y_pred_best = model.predict(val_x)
    val_y_pred_labels_best = tensorflow.argmax(val_y_pred_best, axis=-1)
    val_y_labels_best = tensorflow.argmax(val_y, axis=-1)
    f1_val_best = f1_score(val_y_labels_best, val_y_pred_labels_best, average='weighted')
    print(f'Best Model Validation F1 Score: {f1_val_best:.4f}')
    test_y_pred_best = model.predict(test_x)
    test_y_pred_labels_best = tensorflow.argmax(test_y_pred_best, axis=-1)
    test_y_labels_best = tensorflow.argmax(test_y, axis=-1)
    f1_test_best = f1_score(test_y_labels_best, test_y_pred_labels_best, average='weighted')
    print(f'Best Model Test F1 Score: {f1_test_best:.4f}')
    print('Finished: '+basefilename)
end_time = time.time()
total_time = end_time - start_time
print("Total time taken: ", total_time)
# 获取模型的参数量
model_params = model.count_params()
print(f"Total model parameters: {model_params} parameters")