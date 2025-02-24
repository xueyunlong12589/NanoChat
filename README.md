# NanoChat



本开源项目的目的是想从0开始，将LLM的技术路线跑一遍

包括：数据收集与整理 -> 预训练 ->指令微调 ->强化学习，在新技术出来后，并补充相关内容

主要的是将代码和流程跑通，达到一个可对话的效果，目前不追求更优效果和榜单。

## 一、训练数据

### 1、预训练数据

本项目一共收集了20B tokens左右的数据，包括[中文维基百科](Qwen/Qwen2.5-1.5B-Instruct)和[WuDaoCorpora语料库](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)中随机抽取了部分数据

这两个数据都相对比较干净，因此不做进一步清洗，只对格式进行调整和整理

由于预训练数据一般比较大，所以先对其进行tokenizer之后进行存储，省去训练时分词的时间

`process_wiki.py`和`process_wudao.py`分别对两个数据集进行处理，`merge_data.py`对数据进行合并，整理成一个文件数据

**数据存储格式说明：**

在机器内存有限的情况下，代码中构造`torch.utils.data.dataset`时不能一次读进来，所以进行memmap内存映射，数据存储为.bin二进制文件，另外存储.idx文件用于记录每个句子的起始位置和长度。

如果内存够的话，一次读进来是更快的，可以考虑使用parquet格式存储，可以将文件压缩到较小，并且可以使用索引直接读取每个句子，但是不支持内存映射

### 2、SFT数据

### 3、强化学习数据

## 二、模型架构

本项目的显卡为单机2张A40，资源有限，所以考虑训练小于1B的模型，对于小模型的架构问题，参考了以下几篇资料：

- Rethinking Optimization and Architecture for Tiny Language Models，解读：[https://zhuanlan.zhihu.com/p/681614203](https://zhuanlan.zhihu.com/p/681614203)
- 一篇叫做 [MobileLLM](https://openreview.net/pdf?id=EIGbXbxcUQ) 的 paper，介绍了一些训练小模型的 trick

总结下来，小模型可以优化的点如下：

- 分词器：目前大模型词表数据都达到了上百K，在我们几百M的模型参数下，单单embedding的参数量就占了1/3，留给transformer模型的参数量并不多，可以选择一个小的词表或者自己训练
- 同样参数量下，深而瘦 的模型比 浅而胖 的模型效果更好
- 多轮训练，不同于大模型一般训练一个epoch，小模型可能需要多个epoch
- 其他减少模型大小但相对不失模型性能的结构：tied word embedding、GQA、layer级别的参数共享

由于本项目首先想先快速实现一个模型，所以目前上述因素没有都纳入考虑，最终模型结构参考了qwen2.5-0.5B的结构，具体如下

```
tokenizer：qwen2.5的词表，151644大小
Tied word embedding
ROPE
SwiGLU
RMSNorm
Attention QKV bias
n_layers：24
GQA：14个Q头，2个KV头
dim:896
max_len:512
```

