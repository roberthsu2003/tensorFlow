## tensorFlow介紹
- 包含一個簡單範例

### 什麼是機器學習
-透過資料和結果,建立規則(演算法)
### 傳統程式語言的限制
- 使用判斷和迴圈建立演算法
- 例如自動汽車要有非常龐大的演算法
### 什麼是TensorFlow
- TensorFlow是一個建立機器學習模型的開放平台
- 已經提供很多一般的演算法和樣版供我們使用(節省使用者必需要了解底層數學的運算和邏輯)
- 幫助開發人員突破AI學習的門檻
- 支援開發者部署模型到web,cloud,mobile,embedded
- 建立一個機器學習的流程,稱做training(使用一堆演算法和輸入的資料,以便區分未來輸入的資料)
- 輸入資料至模型,產生出的結果,這個過程稱為inference

#### 使用TensorFlow使用流程
1. 使用已經內建神經網路的評估器,如Keras
2. 提供cpu,gpu或tpu的硬體來做訓練
3. 提供測試用的資料,可以快速透過TensorFlow Data Service 取得學習用的資料
4. 利用TensorFlow Lite,TensorFlow.js部署在不用的系統(web,cloud,mobile,embedded)
### 使用TensorFlow
- 使用vscode,pycharm或colab的IDE開發環境
- 安裝tendorFlow的套件至python

```
#支援CPU
$ pip install tensorflow

#僅支援GPU
$ pip install tensorflow-gpu

#測試版本
import tensorflow as tf
ts.__version__
```




### 使用模型(使用conda-mini)
- 將模型(model.tflite)複制至本機
- python版本使用3.9版
- 安裝tensorflow


```
!python --version

========
Python 3.9.21
```

```bash
pip install tensorflow
```

- **建立模型和訓練模式(created_model)**
- 儲存為tflite
- Y=2X-1(一元一次方程式)

```python
import tensorflow as tf
import numpy as np

# Convert the model to TensorFlow Lite
def convert_to_tflite(model, output_path='model.tflite'):
    """
    Convert a TensorFlow Keras model to TensorFlow Lite format
    
    Args:
        model (tf.keras.Model): The trained Keras model
        output_path (str): Path to save the converted TFLite model
    
    Returns:
        bytes: TFLite model in byte format
    """
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Add optimization techniques
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model to disk
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")

   

# Original model training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]
xs = np.array(x, dtype=float)
ys = np.array(y, dtype=float)

model.fit(xs, ys, epochs=500)

# Convert the model to TFLite
tflite_model_path = 'linear_model.tflite'
convert_to_tflite(model, tflite_model_path)
```


- **使用模型(using_model)

```python
import tensorflow as tf
import numpy as np

def load_and_use_tflite(tflite_model_path):
    """
    Load a TensorFlow Lite model and use it for prediction
    
    Args:
        tflite_model_path (str): Path to the .tflite model file
    
    Returns:
        TFLite Interpreter
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Function to predict using the TFLite model
    def predict(input_data):
        # Prepare input data
        input_data = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
        
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
    
    return predict

tflite_model_path = 'linear_model.tflite'
# Load the TFLite model
tflite_predict = load_and_use_tflite(tflite_model_path)
# Make predictions
test_input = [10.0]
print("TFLite Model Prediction:", tflite_predict(test_input))
```




