同学们好，我今天我要给大家演示的是 Interactive Digital Photomantage. 

这篇文章的整个框架比较清晰，首先通过用户的操作对图片进行分割，然后在梯度域上对图片进行融合。

图片的分割需要用到 Graph Cut, Graph Cuts 的优化目标是这个惩罚函数，它分为两个部分: 数据惩罚和平滑相。



数据项就是我们所说的，当用户指定了一些像素的标签作为标签的时候，我们应该给其他像素多大的惩罚。一个最为简单粗暴的方法是用户选中的像素是低的惩罚，其他一律是高惩罚。

还有一个就是 Designated Color, 我们知道用户选中的像素的颜色，那么我们可以用其他像素和用户选中的像素颜色做比较，如果差距越大，那么惩罚越大。

还有一个是 Luminace, 这个和 Desiganted Color 其实是一样的，只不过 Luminace比对的是亮度而不是颜色。

还有个比较有趣的是 Likelihood, 就是说我们把所有颜色分成比如 20 个类别，然后我们去统计每一个类别中，用户选中的颜色在这个类别里的数量和这个类别里所有的像素个数之比，右边那张图就是一个类似于颜色直方图的，淡蓝色代表属于这个类别但是用户没选中的，深蓝色是属于这个类别并且用户选中了它。我们可以用这个比值作为似然概率作为 Data Penalty.

除此之外还有像 Eraser 这样的迭代的目标， 就是当前这个像素和之前合成的图像之间的区别，这个适用于用户每次增加一个label，我们基于上一次的计算结果来得到数据项，还有很多种方法去定义数据项，我在实验中使用的是最粗暴的第一个。



接下来是平滑项，平滑项也有很多种定义方法，最简单的是两个颜色的欧式距离，



比如我们右上角有两张图片我们对比的是两张图片对应两点的欧氏距离之和。



然后是对梯度的欧氏距离，这和颜色的比较是一样的，除了梯度其实有x和 y这两个梯度方向，所以我们不是比较 RGB 这样三维向量的欧式距离，而是六维向量之间的欧式距离。在实现中我用的是 sobel 算子。



梯度和颜色平滑相，其实是把两个上面两个加起来即可，但是其实梯度一般性幅值要比较小，如果我们不给梯度更大的权重，颜色的影响会比较大。



还有一个是 Edge，这个在论文里说了是 Scalar Edge Potential, 但是我们没查到相关的资料，所以我直接用来 Canny 算法对图像求边缘，然后比较的是两个图像边缘的差。



还有一个 Intuition是，当用户画好几个像素的时候，一般用户会希望这些像素周围的点也是同样的 Label, 所以我们给靠近用户 Label 的像素更低的惩罚，给较远的高惩罚。



接下来一个问题是 Graph Cuts 尽力了，但是如果两个区域照明差距比较大还是没办法完美地融合图像。右边估计钉钉压缩之后看不出来，我发到群里。。大家可以看到在 边缘的地方其实因为亮度的差异会有细微的问题，所以我们需要用 Gradient Domain Fushion去解决。



这是一个 纽曼边界条件问题，我们要求解某一个像素的值，我们希望这个梯度和原图的梯度能保持一致。

要解这个方程我们只要列出像这样的一个线性方程，然后交给共轭梯度法求解就好了。



接下来我给大家演示一下我的实现，顺便也展示一下 GUI。

首先是用户选择不同的图片提供  Label, 然后使用 Match Color 的目标进行计算，为了能及时获得反馈，我使用了低清模式，就是压缩画质，减少迭代次数，从而快速获得结果，这里主要看一下分类的结果的对比：我们可以看到 Match Color 相比之下有严重的问题，它碎片程度非常严重。这也导致了 Gradient Fushion 回天乏术。但是对比 Match Gradient ，它画分的区域非常规则，也让 Graident Fushion非常好做。



接下来我们进入高清模式，因为高清模式计算很慢，所以我快进了很多。



我认为梯度比颜色好有三大原因，一个是它对照明的改变有一定的鲁棒性，除此之外它能更好地捕捉局部性和物体的形状。



在我最初的实现里面，需要3-4分钟才能出结果，性能的瓶颈是 Gradient Domain Fushion, 我们可以对比一下不同迭代次数的图片会发现这么慢的原因是这张图片是从 0 开始迭代的，而我们可以看到，Graph Cuts 得到的结果已经非常好了，所以我们可以直接用 Graph Cuts 的结果作为共轭梯度下降的初始值，这样我们就大大减少了迭代需要的次数，只要 50 次就可以达到之前的3000次的效果，而这个时间的节省是非常客观的。









然后是我的一些想法，目前





低清 Match Color Match Grads, Match Grad & Color, Match 







