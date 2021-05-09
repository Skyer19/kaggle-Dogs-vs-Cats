# kaggle-Dogs-vs-Cats
## 对GPU和CPU的调用
<pre>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
数据集定义
数据集导入
神经网络定义
等
"""
model = Net()          # 实例化一个网络
model = model.to(device)  
</pre>
## 遇到的问题
### 图片的大小不一致
输入到cnn网络进行训练的图片要求图片大小一致，而训练集中的图片大小不一致🤭
解决办法：
<pre>
from PIL import Image

IMAGE_H = 200
IMAGE_W = 200
img = Image.open("图片路径")                   # 打开图片
img = img.resize((IMAGE_H, IMAGE_W))          # 将图片resize成统一大小
</pre>

## torch.utils.data.Dataset类
<pre>
class Dataset(object):
    """
    An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
</pre>
- 通过重载<pre>def __getitem__(self, index):</pre>data.Dataset父类方法，获取数据集中数据内容<br>
- 通过重载<pre>def __add__(self, other):</pre>data.Dataset父类方法，获取数据集的大小<br>

## 卷积神经网络

