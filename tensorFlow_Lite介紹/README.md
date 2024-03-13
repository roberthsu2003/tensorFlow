## TensorFlow Lite 介紹

- 一組完全實作TensorFlow模型的工具
- 檔案盡可能的小,耗電量盡可能低
- 符合不同的平台,web,cloud,android,ios,linux,microcontroller

### 什麼是TensorFlow lite?

- TensorFlow一開始是針對Android,ios系統,設計最佳化的工具

### 建立和轉換模型至TensorFlow Lite

```python
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense


model = Sequential([Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs,ys, epochs=500)

export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)
```


### 轉換模型和使用模型

```python
import tensorflow as tf
import numpy as np


export_dir = 'saved_model/1'
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

import pathlib
tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
to_predict = np.array([[10.0]], dtype=np.float32)
print(to_predict)
interpreter.set_tensor(input_details[0]['index'],to_predict)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)
```