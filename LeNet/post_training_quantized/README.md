#### 技术说明
* post training quantized 属于训练后量化模型


#### 使用说明

* 1， 执行python train.py 训练模型，会直接保存savedModel模型文件
* 2， 执行convert_to_tflite可以将savedModel文件转换成tflite模型，在这里有三种版本的保存，只有调用了converter.post_training_quantize = True 才能亚索模型到1/4


#### pb 和 tflite 预测结果对比
* tflite_predict_v1/2/3.py 使用tflite预测
* checkpoint_predict.py 使用checkpoint文件预测
* 内存大小： checkpoint： 250k   tflite_v1： 250k   tflite_v2： 68k   tflite_v3: 250k
* 性能： checkpoint： 0.9758   tflite_v1： 0.9758   tflite_v2： 0.9755  tflite_v3: 0.9758
* 推断速度(s/1000)： checkpoint： 0.85   tflite_v1： 0.13   tflite_v2： 0.18   tflite_v3: 0.13