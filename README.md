# PDANN
Point-by-point Data-driven Approach for Statistical Downscaling of  Integrated Wave Parameters in Coastal Regions --PDANN的实现代码

![ANN修正](https://github.com/user-attachments/assets/8bf28d39-5764-454a-ace8-56bda85340aa)
PDANN方法中每个子模型的网络结构。

针对代码内容有以下几点温馨提示：
1.因为内存限制，代码中是将大的区域分成八块小区域来处理。如果你的内存显卡允许不许做这个处理。
2.代码的核心逻辑就是寻找高分辨率数据点位周围最近的四个点位的低分辨率数据作为ANN的输入，网络训练很简单，重要的是做好数据预处理以及就近点为的搜索。
3.搜索就近点位的部分代码是根据数据结构特点而设计。
4.更多详细的内容请见论文

