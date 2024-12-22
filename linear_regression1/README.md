# 2元1次方程式(2-variable equation)

y = 2x + 3y - 1

** created_model.py**

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

# Original model training for 2-variable equation z = 2x + 3y - 1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data for z = 2x + 3y - 1
x = [[-1, -1], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
y = [2*(-1) + 3*(-1) - 1, 2*0 + 3*0 - 1, 2*1 + 3*1 - 1, 2*2 + 3*2 - 1, 2*3 + 3*3 - 1, 2*4 + 3*4 - 1]
xs = np.array(x, dtype=float)
ys = np.array(y, dtype=float)

model.fit(xs, ys, epochs=500)

# Convert the model to TFLite
tflite_model_path = 'linear_model_2var.tflite'
convert_to_tflite(model, tflite_model_path)

```

**評估模型**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定義callback類別來收集訓練數據
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.weights1 = []  # 第一個變量的權重
        self.weights2 = []  # 第二個變量的權重
        self.biases = []
        self.losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.layers[0].get_weights()[0]
        self.weights1.append(weights[0][0])
        self.weights2.append(weights[1][0])
        self.biases.append(self.model.layers[0].get_weights()[1][0])
        self.losses.append(logs['loss'])

# 建立模型 (z = 2x + 3y - 1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# 準備訓練資料
x = [[-1, -1], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
y = [2*(-1) + 3*(-1) - 1, 2*0 + 3*0 - 1, 2*1 + 3*1 - 1, 
     2*2 + 3*2 - 1, 2*3 + 3*3 - 1, 2*4 + 3*4 - 1]
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
ax1.set_title('Change in loss value')
ax1.set_xlabel('Training Cycle')
ax1.set_ylabel('Loss value')
ax1.grid(True)

# 繪製權重和偏差變化
ax2.plot(callback.weights1, label='weight (x)')
ax2.plot(callback.weights2, label='weight (y)')
ax2.plot(callback.biases, label='bias')
ax2.set_title('Changes in weights and bias')
ax2.set_xlabel('Training Cycle')
ax2.set_ylabel('Value')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 輸出最終結果
test_input = [[2.0, 3.0]]  # 測試輸入
print("測試輸入:", test_input)
print("預測值:", model.predict(np.array(test_input)))
print("目標方程式: z = 2x + 3y - 1")
print("最終權重:", model.layers[0].get_weights()[0])
print("最終偏差:", model.layers[0].get_weights()[1])

# 計算預測值與實際值的誤差
actual_value = 2*test_input[0][0] + 3*test_input[0][1] - 1
predicted_value = float(model.predict(np.array(test_input)))
print(f"實際值: {actual_value}")
print(f"預測值: {predicted_value}")
print(f"誤差: {abs(actual_value - predicted_value)}")
```

**使用模型using_model1.py**

```python
import tensorflow as tf
import numpy as np

def load_and_use_tflite(tflite_model_path):
    """
    載入雙變量 TensorFlow Lite 模型並進行預測
    
    Args:
        tflite_model_path (str): .tflite 模型檔案路徑
    
    Returns:
        function: 預測函數
    """
    try:
        # 載入 TFLite 模型
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # 獲取輸入和輸出張量資訊
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        def predict(input_data):
            """
            使用模型進行預測
            
            Args:
                input_data (list): 包含兩個輸入值的列表 [x1, x2]
            
            Returns:
                np.array: 預測結果
            """
            if len(input_data) != 2:
                raise ValueError("需要exactly兩個輸入值")
                
            # 準備輸入數據
            input_array = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
            
            # 設置輸入張量
            interpreter.set_tensor(input_details[0]['index'], input_array)
            
            # 執行推論
            interpreter.invoke()
            
            # 獲取輸出張量
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            return output_data
        
        return predict
        
    except Exception as e:
        print(f"模型載入錯誤: {str(e)}")
        return None

if __name__ == "__main__":
    # 載入模型
    model_path = 'linear_model_2var.tflite'
    predict_fn = load_and_use_tflite(model_path)
    
    if predict_fn is not None:
        # 測試案例
        test_inputs = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0]
        ]
        
        for test_input in test_inputs:
            try:
                result = predict_fn(test_input)
                print(f"輸入: {test_input}, 預測結果: {result[0][0]}")
            except Exception as e:
                print(f"預測錯誤: {str(e)}")
```