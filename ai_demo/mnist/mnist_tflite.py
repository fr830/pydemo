
'''

需要使用 tensorflow 1.13.0

'''


import tensorflow as tf
from PIL import Image
import numpy as np

model_path = "F:/demo/py/pydemo/ai_demo/mnist/model/mnist.tflite"
image_file = 'F:/demo/py/pydemo/ai_demo/mnist/images/0.jpg'

print('tensorflow version :', tf.__version__)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print('input tensors :', str(input_details))
output_details = interpreter.get_output_details()
print('output tensors :', str(output_details))

image = Image.open(image_file)
img_data = np.array(image, dtype='float32').reshape(1, 784)

interpreter.set_tensor(input_details[0]['index'], img_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

result = np.squeeze(output_data)
print('recognition result:{}'.format(result))