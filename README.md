# NanoChat



本开源项目的目的是想从0开始，将LLM的技术路线跑一遍

包括：数据收集与整理 -> 预训练 ->指令微调 ->强化学习，在新技术出来后，并补充相关内容

主要的是将代码和流程跑通，达到一个可对话的效果，目前不追求更优效果和榜单。

## 一、训练数据

### 1、预训练数据

本项目一共收集了50B tokens左右的数据，包括[中文维基百科](Qwen/Qwen2.5-1.5B-Instruct)和[WuDaoCorpora语料库](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)中随机抽取了部分数据

| 中文预训练语料                                               | 描述                   |
| ------------------------------------------------------------ | ---------------------- |
| [中文维基百科](Qwen/Qwen2.5-1.5B-Instruct)                   | 中文Wikipedia的数据    |
| [WuDaoCorpora语料库](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) | 中文悟道开源的200G数据 |

这两个数据都相对比较干净，因此不做进一步清洗，只对格式进行调整和整理

由于预训练数据一般比较大，所以先对其进行tokenizer之后进行存储，省去训练时分词的时间

`process_wiki.py`和`process_wudao.py`分别对两个数据集进行处理，`merge_data.py`对数据进行合并，整理成一个文件数据

**数据存储格式说明：**

在机器内存有限的情况下，代码中构造`torch.utils.data.dataset`时不能一次读进来，所以进行memmap内存映射，数据存储为.bin二进制文件，另外存储.idx文件用于记录每个句子的起始位置和长度。

如果内存够的话，一次读进来是更快的，可以考虑使用parquet格式存储，可以将文件压缩到较小，并且可以使用索引直接读取每个句子，但是不支持内存映射

### 2、SFT数据

| SFT语料                                                      | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [identity中文自我认知数据](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) | 使用了llamafactory提供的模型自我认知数据的中文部分，大概75条数据 |
| [BELLE 2M](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | 源自BelleGroup的一部分SFT数据，大概200W条                    |
| [Alpaca-gpt4-zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | 参考Alpaca方法基于GPT4得到的self-instruct数据，约5W条，llamafactory去掉了6103条错误的数据 |

SFT的语料一般较小，没有必要提前分词，而是在构建Dataloader的时候进行分词。

### 3、强化学习数据



## 二、模型架构

本项目的显卡为单机2张A40，资源有限，所以考虑训练小于1B的模型，对于小模型的架构问题，可参考以下几篇资料：

- Rethinking Optimization and Architecture for Tiny Language Models，解读：[https://zhuanlan.zhihu.com/p/681614203](https://zhuanlan.zhihu.com/p/681614203)
- 一篇叫做 [MobileLLM](https://openreview.net/pdf?id=EIGbXbxcUQ) 的 paper，介绍了一些训练小模型的 trick

总结下来，小模型可以优化的点如下：

- 分词器：目前大模型词表数据都达到了上百K，在我们几百M的模型参数下，单单embedding的参数量就占了1/3，留给transformer模型的参数量并不多，可以选择一个小的词表或者自己训练
- 同样参数量下，深而瘦 的模型比 浅而胖 的模型效果更好
- 多轮训练，不同于大模型一般训练一个epoch，小模型可能需要多个epoch
- 其他减少模型大小但相对不失模型性能的结构：tied word embedding、GQA、layer级别的参数共享

由于本项目首先想先快速实现一个模型，所以目前上述因素没有都纳入考虑，具体如下

```
tokenizer：qwen2.5的词表，151650大小
Tied word embedding
ROPE
SwiGLU
RMSNorm
Attention QKV bias
```

训练模型具体参数如下所示：

| 模型名称      | 模型参数                                                     |
| ------------- | ------------------------------------------------------------ |
| NanoChat-0.3B | max_len=512<br />dim=512<br />n_layers=24<br />v_head=14<br />kv_head=2 |

## 三、模型训练

对于小模型来说，deepspeed已经足够了，训练部分代码都是了accelerate库来简化了分布式训练代码，并集成了deepspeed，支持了断点续训

```
单卡训练：python train_pretrain.py
多卡训练DDP：accelerate launch train_pretrain.py
多卡训练DeepSpeed：accelerate launch --config_file ds_zero2.yaml train_pretrain.py
```

## 四、模型权重

NanoChat-0.3B-base（[huggingface](https://huggingface.co/xueyunlong/NanoChat-0.3B-base/tree/main)|[modelscope](https://modelscope.cn/models/xueyunlong/NanoChat-0.3B-base/files))

NanoChat-0.3B-instruction（[huggingface](https://huggingface.co/xueyunlong/NanoChat-0.3B-instruct/tree/main)|[modelscope](https://modelscope.cn/models/xueyunlong/NanoChat-0.3B-instruct/files)）

## 五、模型测试结果
