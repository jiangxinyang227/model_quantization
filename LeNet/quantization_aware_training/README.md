#### 技术说明
* quantization aware training 属于训练中的量化，一般来说训练时模型先迭代一定步数到模型基本收敛，然后引入伪量化训练一定步数到模型最终收敛，预测时需要将伪量化的模型转换成完全量化的模型。


#### 使用说明

* 1， 执行python train.py 训练模型，训练时使用伪量化，需要将tf.contrib.quantize.create_training_graph函数放置在loss计算后面，定义优化器前面执行
* 2， 训练完之后保存为checkpoint 文件
* 3， 执行freeze.py 将checkpoint文件转换成freeze pb文件，但是要注意在这里要重新定义一个graph，然后再加载训练时保存的变量值到图中。
* 4， 在转换成freeze pb文件时会将优化器中的临时参数去掉，最常见的就是adam算法中的一阶矩和二阶矩中间值，所以可以看到freeze pb文件的内存占用会减小2/3
* 5， 执行convert_to_tflite可以将freeze pb文件转换成tflite模型
* 6， read_node.py 可以查看pb文件中有哪些node，这样在convert to tflite的时候指导输入node和输出node有哪些


#### pb 和 tflite 预测结果对比
* tflite_predict.py 使用tflite预测
* freeze_pb_predict.py 使用freeze pb文件预测
* 内存大小： pb：258k   tflite： 66k
* 性能： pb：0.9777   tflite：0.9553
* 推断速度(s/1000)： pb：0.5   tflite：0.15