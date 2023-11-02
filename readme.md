# 1205 update:

## 当前策略：

1. 数据读取

2. 数据预处理

   * 清洗`nan`值

   * 对训练集做`MinMax`标准化

   * 形成训练序列（将数据规整成两列，一列作为训练数据，一列为对应的标签）

     ![image-20221205164937357](https://github.com/Anna-fence/Pictures/blob/master/image-20221205164937357.png)

3. 训练模型并保存

4. 测试模型

   和上图差不多，绿色框框实际就是测试时的输入序列，蓝色框框既是模型的输出又是对应的标签位置。

## 有疑问的地方

1. `nan`应该怎么处理呢？现在是直接置为`0`的，但是显然不太合理。或者用插值填一个数进去？总觉得这样会让模型学到不该学的东西，同时如果直接去掉的话也不合理，因为这个时间序列就少了一个值，时刻就对不上了。
2. 数据归一化是否有必要？从目前的结果来看，归一化似乎让预测出来的数据波动没有那么大了，很可能是这个过程中失去了精度。
3. 事实上`nan`的处理方式会直接影响到归一化的效果。
4. 现在的预测策略是否合理？

## 贴一点结果上来

### `learning rate = 0.0001`

`epoch = 30`

![image-20221205174623111](https://github.com/Anna-fence/Pictures/blob/master/image-20221205174623111.png)

`epoch = 50`

![image-20221205171827637](https://github.com/Anna-fence/Pictures/blob/master/image-20221205171827637.png)

`epoch = 80`

![image-20221205172509950](https://github.com/Anna-fence/Pictures/blob/master/image-20221205172509950.png)

`epoch = 100`

![image-20221205171848560](https://github.com/Anna-fence/Pictures/blob/master/image-20221205171848560.png)

（感觉效果都不是很好，但是看不懂调参改往哪里调，这是过拟合了还是啥）
