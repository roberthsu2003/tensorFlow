## tensorFlow介紹


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






