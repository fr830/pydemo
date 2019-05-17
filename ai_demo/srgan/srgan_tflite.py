
'''

tflite 的调用目前只能在tensorflow 1.13.0版本下   conda py36tf

bazel-bin/tensorflow/lite/toco/toco \
  --input_file=/home/antutu/ai/tensorflow_models/srgan.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=/home/antutu/ai/tensorflow_models/srgan.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=inputs_raw \
  --input_shapes=1,120,125,3 \
  --output_arrays=convert_image/convert_image

'''

import tensorflow as tf
import numpy as np
import scipy.io
import cv2
import scipy.misc as sic

tflite_file = './models/srgan_120_125.tflite'
image_lr_file = './images/img_001.png'

print('tensorflow version :', tf.__version__)

interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

# tensors = interpreter.get_tensor_details()
# for item in tensors:
#     print(item['name'])

#tf.lite.TocoConverter()

input_details = interpreter.get_input_details()
print('input tensors :', str(input_details))
output_details = interpreter.get_output_details()
print('output tensors :', str(output_details))

image_lr_data = sic.imread(image_lr_file, mode="RGB").astype(np.float32)
image_lr_data = image_lr_data / np.max(image_lr_data)
input_im = np.array([image_lr_data]).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], np.resize(input_im, (1, input_im.shape[1], input_im.shape[2], 3)).astype('float32'))

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

nn = 0