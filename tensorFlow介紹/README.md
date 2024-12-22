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

#### NumPy 陣列轉換與模型訓練說明

這段程式碼展示了機器學習模型訓練和轉換的關鍵步驟，讓我們逐步解析：

**資料準備**

```
x = [-1, 0, 1, 2, 3, 4] #輸入資料x
y = [-3, -1, 1, 3, 5, 7] #目標資料y
```

**轉換為 NumPy 陣列：**
 
```python
xs = np.array(x, dtype=float)
ys = np.array(y, dtype=float)

# 這個轉換很重要，因為：
# TensorFlow 需要 NumPy 陣列格式的數據
# dtype=float 確保數據型別為浮點數，避免精度問題 
# 統一的數據格式有助於提高計算效率
```


#### 模型訓練

```python
model.fit(xs, ys, epochs=500) 

#fit()方法是 TensorFlow 中啟動訓練的標準方式
#epochs=500 表示模型將重覆訓練 500 次
#訓練過程中模型會不斷調整權重以最小化損失函數
```


#### 觀察權重和損失函數
前5筆損失函數
![](./images/pic1.png)

最後5筆損失函數
![](./images/pic2.png)

**取得最後權重(weight)和最後偏差值(bias)**

```python
import tensorflow as tf
import numpy as np

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]
xs = np.array(x, dtype=float)
ys = np.array(y, dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print("權重:", l0.get_weights()[0], "偏差:", l0.get_weights()[1])

#=====output======
[[18.9838]]
權重: [[1.9976522]] 偏差: [-0.9927207]
```

#### 評估模型(evaluation_models.ipynb)
##### 計畫步驟：

1. 建立callback函數來收集訓練過程的權重和損失值
2. 修改訓練過程以使用callback函數
3. 使用matplotlib繪製訓練過程的圖表
   - 繪製損失值變化
   - 繪製權重變化
4. 圖表展示在不同的子圖中

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定義callback類別來收集訓練數據
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.weights = []
        self.biases = []
        self.losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.layers[0].get_weights()[0][0][0])
        self.biases.append(self.model.layers[0].get_weights()[1][0])
        self.losses.append(logs['loss'])

# 建立模型
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

# 準備訓練資料
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]
xs = np.array(x, dtype=float)
ys = np.array(y, dtype=float)

# 建立回調實例
callback = TrainingCallback()

# 訓練模型
history = model.fit(xs, ys, epochs=500, callbacks=[callback])

# 創建圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 繪製損失值變化
ax1.plot(callback.losses)
ax1.set_title('Change in loss value')# 損失值變化
ax1.set_xlabel('Training Cycle') # 訓練週期
ax1.set_ylabel('loss value') # 損失值
ax1.grid(True)

# 繪製權重和偏差變化
ax2.plot(callback.weights, label='weights')
ax2.plot(callback.biases, label='biases')
ax2.set_title('Changes in weights and biases') # 權重和偏差變化
ax2.set_xlabel('Training Cycle') # 訓練週期
ax2.set_ylabel('numerical value') # 數值
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 輸出最終結果
print("預測值:", model.predict(np.array([10.0])))
print("最終權重:", l0.get_weights()[0])
print("最終偏差:", l0.get_weights()[1])

#====output====
預測值: [[18.976908]]
最終權重: [[1.9966532]]
最終偏差: [-0.989624]
```

![](./images/output.png)

#### 將模型轉換為TFLite 
最後，將訓練好的模型轉換為 TFLite 格式：
- TFLite 是專為移動和嵌入式設備優化的格式
- 轉換後的模型體積更小，運行更快
- 儲存為 `.tflite` 檔案方便部署到其他平台

```python
def convert_to_tflite(model, output_path='model.tflite'):
    """
    轉換TensorFlow Keras model 成為Tensorflow lite 格式
    參數:
        model (tf.keras.Model):訓練完成的keras model
        output_path (str): 要儲存為tensorflow lite格式的路徑和檔案名稱(tflite)    
    Returns:
        None
    """
    # 建立converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Add optimization techniques
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model to disk
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
tflite_model_path = 'linear_model.tflite'
convert_to_tflite(model, tflite_model_path)


```


#### 最終訓練,儲存為linear_model.tflite

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


- **使用模型(using_model)**

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




