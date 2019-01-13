
首先训练: 进入sample.py程序，将需要的训练集2个文件把那两个文件拷贝在2048big homework的根目录下面，调参, 设置batchsize 为512。设置好后直接运行sample.py? 会生成model.h5文件 即权重


然后是如何获得训练集， 定义一个全集变量2个数组，在助教给的2048 agent里面有每一步的循环，进入棋盘，每到一步将棋盘和方向都给保存。获取训练集要用expectMaxAgent

最后运行evluate.py即可
