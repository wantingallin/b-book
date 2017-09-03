# b-book
### 2. Pytorch基本语法

#### 2.1 张量（Tensor）
Pytorch的运算单元叫做Tensor（张量）。我们可以将张量理解为一个多维数组，一阶张量即为一维数组，通常叫做向量（vector）；二阶张量即为二维数组，通常叫做矩阵（matrix）；三阶张量即为三维数组；n阶张量即为n维数组，有n个下标。
![alt](https://images-cdn.shimo.im/aWDUNHOuvVsIJMlF/%E5%9B%BE%E7%89%8728.png!thumbnail)


首先，让我们看看该如何定义张量。

首先导入Pytorch的包：（如果没有安装的话，请看安装教程）

```python
import torch
```

创建一个5*3的二阶张量，随机矩阵：


``` Python
x=torch.rand(5,3) 
```

创建一个5*3的全是1的张量，全是1的矩阵：

```Python
x=torch.ones(5,3) 
```

创建一个5*3的全是0的张量，全是0的矩阵：


```Python
x=torch.zeros(5,3) 
```

Numpy中的array也是一个n维数组，那么Numpy中的array和Pytorch中的tensor有什么区别呢? 最大的不同在于Pytorch中的tensor是可以在GPU中计算的，这大大提高了运算速度。
那Pytorch是如何在GPU上使用tensor呢？通常的做法是需要判断你的计算机是否能够使用GPU，判断语句为：


```Python
torch.cuda.is_available()
```

如果输出是yes，则表明可以将tensor放到GPU上面进行运算，则只需要将你定义的tenor转化到GPU上即可，例如，将x,y放到GPU上，则你只需要

```Python
x=x.cuda()
y=y.cuda() 
print(x+y)
```

当然也有可能你的计算机版本设置不支持GPU，这种情况下可以考虑在云服务器平台上面运行，如FloydHub平台，https://www.floydhub.com/。 

在Pytorch中提供了numpy和torch.tensor之间的简单转化语句： 

从numpy到tensor的转化：

```Python
torch.from_numpy(a)
``` 

从tensor到numpy的转化：

```Python
a.numpy()
```
。 
首先生成torch.tensor与numpy 的array

```Python
x_tensor=torch.randn(2,3) 
y_numpy=np.random.randn(2,3)
```

将tensor转化为numpy：

```Python
x_numpy=x_tensor.numpy()
```


将numpy转化为tensor：

```Python
y_tensor=torch.from_numpy(x_torch)
```

张量的加法运算：（前提是张量的尺寸一样，矩阵一样，符合矩阵加法运算）
```Python
x=torch.zeros(5,3)  
y=torch.ones(5,3)  
z=x+y
```

张量的乘法（mm：matrix multiple）要符合矩阵运算的法则：

```Python
q=x.mm(y.t)
```


其中y.t表示矩阵y的转置 
张量的基本运算语法有很多，包括换位、索引、切片、数学运算、线性算法和随机数等等。详见：torch - Pytorch 0.1.9 documentation。
具体介绍Pytorch中几个基本的张量运算。 

#### 动态计算图（Dynamic Computation Graph）
我们知道，人工神经网络之所以能够在诸多机器学习算法中生成成为一个迄今为止最主流的一种学习算法，就是因为它可以利用反向传播算法来更新它内在的每一个计算单元。由于通过反向传播算法我们能够精确地计算出网络中每一个单元对于网络表现的贡献（这就是所谓的梯度信息），这种技术使得神经网络在学习训练的时候就具备非常高的效率，它可以精准地将预测错误精准地定位到每一个系统个体单元，从而避免大量无效率的学习。

在深度学习框架出现之前，人们需要针对每一种不同的神经网络架构编写不同的反向传播算法，这就使得我们构造神经网络系统的难度非常大。现在，有了计算图技术我们可以不用为每一种不同架构的网络定制不同的反向传播算法，我们只需要关注我如何实现神经网络的前馈过程的运算步骤，那么当我们搭建完整个系统之后，我们就可以让反向传播算法自动进行。这大大提高了人们打开神经计算系统的效率。

传统的深度学习框架，例如TensorFlow、Theano等都采用了静态计算图技术。Pytorch相比与Tensorflow来说，最大的特点是它可以动态地构建计算图。这就使得我们构造计算图更加简便、容易，而且很容易进行调试、追踪。

那么，究竟什么是计算图（Computational Graph）呢？它实际上是一种描述、记录Tensor的运算过程的抽象图模型。一张计算图包括两类节点，分别是变量（Variable）和运算（Computation）。图上的有向连边表示的是运算上的一种前后顺序。当我们在PyTorch中用自动微分型变量（Autograd）进行运算的时候，无论这个计算过程多么地复杂，系统都会自动地构造一张计算图来记录所有的运算过程。在构建好动态计算图之后，我们可以非常方便地利用“.backward()”函数来自动地进行反向传播算法，从而计算每一个变量的梯度信息。而这一切的实现都需要自动微分变量的支持。


自动微分变量（Atuograd.Variable）
为了解释动态计算图，需要先了解一个重要的命令，叫做自动微分变量（atuograd.Variable）这是一种新的数据结构，Variable 和tensor有什么区别呢？tensor只是一种张量的数据形式，可以进行多种运算，但是无法构建计算图。而对于Variable的所有运算都可以自动构建计算图。

它是怎么做到的呢？Variable包含了三个重要的参数，如下图：
![alt](https://images-cdn.shimo.im/yJONQ2Ns0JcfQKX4/%E5%9B%BE%E7%89%8729.png!thumbnail)


变量Variable不仅保存了tensor形式的数据data（即计算图中的节点），还保存产生这个Variable的计算，通过.grad_fn（老版本是creator）我们可以查看是哪个运算导致了现在这个Variable的出现。另外，每个Variable还有.grad用于存储variable的梯度值。当正向运算结束之后，在反向传播阶段，我们只需要通过调用”.backward “，就可以计算反向传播的梯度信息，并将叶节点的导数值存储在”.grad“中。
动态计算图实例演示
让我们举一个小例子，比如我们通过Pytorch构建y=x+2的计算图，对应的Pytorch语句为：
首先导入Pytorch中自动微分变量

```Python
from torch.autograd import Variable
```

创建一个叶节点Variable，包裹了一个2*2张量x，将需要计算梯度属性置为True

```Python
x = Variable(torch.ones(2, 2), requires_grad=True)  
```

输出x,x是一个Variable：
```python
Variable containing:
  1  1  1  1
[torch.FloatTensor of size 2x2]
```



进行函数运算,得到Variable y

y = x + 2 
通过y.grad_fn查看Variable的grad_fn属性：
<torch.autograd._functions.basic_ops.AddConstant at 0x11714fba8>
由上述表示可知.grad_fn储存的是运算（basic_ops.AddConstant）的信息，AddConstant表示函数进行的运算时加上了常数。

在执行y=x+2的过程中，系统已经开始自动构建了动态计算图，如下图所示：

![alt](https://images-cdn.shimo.im/83eILfbcL8MV7ds5/%E5%9B%BE%E7%89%8733.png!thumbnail)


接着，我们再进行函数运算z=y∗y,得到variable z
z = y * y
通过z.grad_fn查看Variable z的grad_fn属性
<torch.autograd._functions.basic_ops.Mul at 0x1167f7048>
由上述表示可知.grad_fn储存的是函数的信息，.Mul表示函数进行的运算是乘法。此时，动态计算图更新为：

![alt](https://images-cdn.shimo.im/BobSdfncpnwcHlLC/%E5%9B%BE%E7%89%8734.png!thumbnail)

接下来，让我们看看z中的计算结果:
z.data 
用.data返回一个Variable所包裹的Tensor。 它的输出信息为：


### 代写

此时，假设我们完成了整个运算过程的构建。至此，我们知道，整个计算过程实际上完成了一个符合函数的构建：
z=y^2=(x+2)^2

这实际上就是一个多层的广义神经网络。最后的动态计算图为：

![alt](https://images-cdn.shimo.im/QGlKEtH9ZLkRsnrX/%E5%9B%BE%E7%89%8732.png!thumbnail)


接下来，我们可以希望知道，如果x发生了一些小的变化\delta X，z会发生多大的变化；或者反过来，如果我们观察到了z的小变化\delta z，那么它是由x多大的变化\delta x所引起的呢？这相当于要计算导数：

\partial \delta z  /  \partial \delta x

通过.backward来进行梯度的反向传播，我们可以得到这个导数信息：
z.backward()  
之后，我们可以用z.grad来进行查看叶节点的梯度信息，

```Python
print(z.grad)
print(y.grad)
print(x.grad)
```

注意，由于z和y都不是叶子节点，所以都没有梯度信息。

得到的输出应该是：

```Python
NoneNoneVariable containing:
  1.5000  1.5000  1.5000  1.5000
[torch.FloatTensor of size 2x2]
```

所以，梯度的计算可以自动化地进行，非常方便。无论函数依赖关系多么复杂，也无论神经网络有多深，我们都可以通过backward来完成梯度的自动计算，这就是动态计算图的优势。

为了进一步理解backward()的厉害所在，也为了进一步理解动态计算图，让我们再来看一个例子：
首先创建一个1*2的Variable（1维向量）s
```Python
s = Variable(torch.FloatTensor([[0.01, 0.02]]), requires_grad = True) 
```

创建一个2*2的矩阵型x

```Python
x = Variable(torch.ones(2, 2), requires_grad = True)
```
反复用s乘以x（矩阵乘法），注意s始终是variable

```Python
for i in range(10):
    s = s.mm(x)  是1*2的Variable
    
```

对s中的各个元素求均值，得到一个1*1的scalar（标量，即1*1张量）

```Python
z = torch.mean(s) 
```



这个过程的动态计算图为：


![alt](


同样，我们可以很轻松地计算叶节点变量的梯度信息：

```Python
z.backward() 
print(x.grad)  
print(s.grad) 
```


注意事项：
1，只有叶节点才能计算grad信息，非叶节点不能计算。这是因为非叶节点大多都是计算单元。而有些非叶节点并不能对计算的结果造成影响，因此也不能计算梯度信息。

2，使用backward()计算每个叶节点的梯度信息的时候，梯度是累加的，所以在调用此函数之前需要将叶节点的梯度清零。


