
'''

tflite 的调用目前只能在tensorflow 1.13.0版本下

bazel-bin/tensorflow/lite/toco/toco \
  --input_file=/home/antutu/ai/tensorflow_models/vdsr.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=/home/antutu/ai/tensorflow_models/vdsr.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=input \
  --input_shapes=1,300,300,1 \
  --output_arrays=shared_model/Add

'''

import tensorflow as tf
import numpy as np
import scipy.io
import cv2

pb_file = 'F:/demo/py/pydemo/ai_demo/vdsr/models/vdsr.tflite'
mat_file = 'F:/demo/py/pydemo/ai_demo/vdsr/images/0_2.mat'

print('tensorflow version :', tf.__version__)

interpreter = tf.lite.Interpreter(model_path=pb_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print('input tensors :', str(input_details))
output_details = interpreter.get_output_details()
print('output tensors :', str(output_details))

mat_dict = scipy.io.loadmat(mat_file)
input_data = mat_dict["img_2"]

interpreter.set_tensor(input_details[0]['index'], np.resize(input_data, (1, input_data.shape[0], input_data.shape[1], 1)).astype('float32'))

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = np.resize(output_data, (input_data.shape[0], input_data.shape[1]))

cv2.imshow("input", input_data)
cv2.imshow("vdsr", output_data)
cv2.waitKey()