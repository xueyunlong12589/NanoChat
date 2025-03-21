# NanoChat



本开源项目的目的是想从0开始，将LLM的技术路线跑一遍

包括：数据收集与整理 -> 预训练 ->指令微调 ->强化学习，在新技术出来后，并补充相关内容

主要的是将代码和流程跑通，达到一个可对话的效果，目前不追求更优效果和榜单。

------

### 更新日志

2025.3.13  模型支持了deepseek的MLA和MOE，模型加紧训练中

------

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

| DPO语料                                                      | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [UltraFeedback-chinese](https://huggingface.co/datasets/opencsg/UltraFeedback-chinese) | UltraFeedback Chinese数据集 UltraFeedback-Chinese 是根据UltraFeedback数据集的构建方法制定的中文版本，专为训练强大的奖励模型和批评模型而设计 |



## 二、模型架构

本项目的显卡为单机2张A40，资源有限，所以考虑训练小于1B的模型，对于小模型的架构问题，可参考以下几篇资料：

- Rethinking Optimization and Architecture for Tiny Language Models，解读：[https://zhuanlan.zhihu.com/p/681614203](https://zhuanlan.zhihu.com/p/681614203)
- 一篇叫做 [MobileLLM](https://openreview.net/pdf?id=EIGbXbxcUQ) 的 paper，介绍了一些训练小模型的 trick

总结下来，小模型可以优化的点如下：

- 分词器：目前大模型词表数据都达到了上百K，在我们几百M的模型参数下，单单embedding的参数量就占了1/3甚至一半，留给transformer模型的参数量并不多，可以选择一个小的词表或者自己训练
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

NanoChat-0.3B-sft（[huggingface](https://huggingface.co/xueyunlong/NanoChat-0.3B-SFT)|[modelscope](https://modelscope.cn/models/xueyunlong/NanoChat-0.3B-SFT)）（没有经过强化学习）

NanoChat-0.3B-instruct（[huggingface](https://huggingface.co/xueyunlong/NanoChat-0.3B-instruct)|[modelscope](https://modelscope.cn/models/xueyunlong/NanoChat-0.3B-instruct)）（经过强化学习）

## 五、模型测试结果

将模型与一般大小的qwen2.5-0.5B做对照

**base模型**

```
#示例一：Input：'床前明月光，疑是地上霜。举头望明月，'
NanoChat-0.3B-base：'低头思故乡。 《诗经·小雅·伐木》：“伐木知已生，只影向何方？” 毛泽东《论持久战》：“如果敌人在战斗中，把敌人从睡梦中叫醒，那么我们就要用枕头和被子来掩护自己。”。基本信息 【词目】床前明月光 【拼音】y'
Qwen-0.5B-base：'低头思故乡\n《静夜思》是唐代诗人李白创作的一首五言绝句，描绘了作者在床前观看明亮的明月，想起屋外的夜晚，感到月亮已从地平线上升起，仿佛在照耀着地上的霜，诗人抬头仰望明月，不禁低头向心上人诉说心中的思念，表达了诗人无尽的乡愁。\n以下是对这首诗的简要分析：\n一、明月皎洁，营造 '

#示例二：Input：'从前有座山，山里有座庙，'
NanoChat-0.3B-base：'庙里有个和尚。和尚说：“我这一生最大的愿望就是，能和你一起在山下修行。” 于是，和尚就每天早晚朝拜在山下。有一天，他看见一个和尚在山上修行，和尚问：“你为什么天天朝拜在山下？”和尚说：“因为我是佛祖啊！”和尚又问：“你为什么天天朝拜在山下？”和尚说：“因为我是佛祖'
Qwen-0.5B-base：'庙里住着个神仙，神仙倒有神仙话。他讲了一个好听的故事，叫《小矮人的七日生活》，教给孩子们做菜、洗头、听故事、讲故事。小矮人每天在太阳下走两圈，这中间会掉下一块石头和一朵云彩。这朵云彩一到，他就起床洗手洗澡，又来到池塘，又来到山那边的地底下，又来到海边。这天晚上，他要'
```

base模型qwen具有更多连续性，事实性依据也更多。NanoChat虽然也能续写，但是容易胡编乱造且容易出现复读机现象

**sft和instruct模型**

```
#示例一：Input：'12+21等于几？'
NanoChat-0.3B-sft：'12 + 21 = 45。\n这是一个简单的数学问题，其中加法是两个数字相加的运算符。在这种情况下，我们有两个数字：12和21。我们将这两个数字相加，并将结果存储在变量“result”中。然后，我们将结果输出到屏幕上，以便我们可以看到答案。\n因此，12 + 21 = 45。\nassistant\n这个问题的答案是45。 \n要理解这个答案，我们需要知道如何执行加法'
NanoChat-0.3B-instrcut：'12 + 21 的计算方法如下：\n1. 首先，将两个数的个位相加，即 3 + 4 = 7。\n2. 然后，将进位（在本例中为 0）添加到十位上，即 1 + 9 = 21。\n3. 接下来，将百位上的数字相加，即 1 + 8 = 21。\n4. 最后，将进位'
Qwen-0.5B-instruct：'12 + 21 等于 33。'

#示例二：Input：'万有引力是谁提出的。'
NanoChat-0.3B-sft：'万有引力是由英国物理学家艾萨克·牛顿在17世纪提出的一种物理学理论，它描述了物体之间的相互作用力。\n牛顿的万有引力定律表明，任何两个物体之间都存在一个质量为m的物体对另一个物体施加的作用力，这个力的大小与它们的质量成正比，与它们之间的距离的平方成反比。换句话说，如果一个物体的质量越大，它的引力就越大；反之，如果它的质量越小，它的引力就越小。\n牛'
NanoChat-0.3B-instrcut：'万有引力是牛顿在1687年提出的一个基本的物理学理论，它描述了物体之间相互作用的力量。\n以下是对万有引力的详细解释：\n### 1. 引力的定义\n万有引力是由两个物体之间的引力产生的。根据牛顿第二定律，任何两个物体之间的引力与它们的质量成正比，与它们之间的距离的平方成反比。具体来说，质量越大，引力就越大；而距离越远，则引力就越小。\n### 2.'
Qwen-0.5B-instruct：'万有引力是牛顿在1687年提出的一个物理定律，这个定律描述了两个物体之间的相互吸引力。\n\n牛顿的万有引力理论基于以下三个基本假设：\n1. 物体间存在相互吸引的力。\n2. 这个力与它们的质量成正比，并且与它们之间距离的平方成反比。\n3. 任何两个质量不同的物体都会相互吸引。\n\n根据这些假设，牛顿提出了著名的万有引力定律'

#示例三：Input：'天空为什么是蓝色的。'
NanoChat-0.3B-sft：'天空之所以呈现蓝色，是因为大气层中的气体分子散射了太阳光中的短波长颜色（如红色、橙色和黄色）。\n当太阳光穿过大气层时，它会与空气中的氧气、氮气和其他微小颗粒发生碰撞，并将这些光线分散到不同的方向上。其中，蓝色光波长较短，因此更容易被散射。而其他颜色的光波长较长，因此更容易被分散。\n在日落或日出时，太阳光经过更长的大气'
NanoChat-0.3B-instrcut：'天空之所以呈现出蓝色的颜色，是因为大气层对太阳光的散射作用。\n \n1. 大气层：大气层是由气体、微粒和空气分子组成的，这些物质在进入地球的大气层时会与大气中的气体发生相互作用，其中最常见的是氮气（N2）、氧气（O2）和氩气（Ar）。当这些气体分子与大气层中的气体分子相互作用时，它们会将光线分散到不同的方向上，从而导致我们看到的天空呈现蓝色'
Qwen-0.5B-instruct：'天空之所以呈现为蓝色，主要是因为大气中的气体和颗粒物在阳光照射下吸收了太阳光中的短波长光线（如蓝光），而反射出了较长波长的光线（如红、橙、黄等）。\n\n具体来说，当太阳光穿过地球的大气层时，它会与空气中的水蒸气、二氧化碳和其他微小颗粒物发生相互作用。这些物质对不同颜色的光有不同的吸收能力。例如，蓝光比红光'

#示例四：Input：'老年人如何预防骨质疏松？'
NanoChat-0.3B-sft：'老年人的骨骼健康非常重要，因为他们的身体需要更多的钙来支持其生长和发育。然而，老年人的骨骼健康状况可能受到多种因素的影响，包括生活方式、饮食习惯、运动量等。\n以下是一些预防老年人骨质疏松的方法：\n1. 增加钙摄入量：老年人通常会流失大量的钙，因此增加每天摄入足够的钙是非常重要的。可以通过饮用牛奶、酸奶、豆奶等食物来获得足够的钙。\n2. 饮食均衡：老年人应该注意饮食'
NanoChat-0.3B-instrcut：'老年人的骨骼健康是至关重要的，因为随着年龄的增长，他们的骨骼逐渐失去弹性、脆弱和易碎性。以下是一些预防骨质疏松的方法：\n### 1. 预防骨折： \n1.1 坚持定期锻炼：有规律的运动可以增强骨骼密度，减少骨质流失的风险。建议每周至少进行150分钟的中等强度有氧运动，如快走、跑步或游泳。\n1.2 锻炼时选择合适的运动方式'
Qwen-0.5B-instruct：'老年人预防骨质疏松，可以采取以下措施：\n\n1. 增加钙的摄入：钙是骨骼的主要成分之一。可以通过食物（如牛奶、奶酪、豆腐等）或补充剂来增加钙的摄入。\n\n2. 保持适当的体重：过重会增加骨密度，导致骨质流失。因此，保持健康的体重对于预防骨质疏松非常重要。\n\n3. 定期进行体育锻炼：定期进行有氧运动和力量训练可以帮助'
```

**NanoChat的sft和instruct对比：**

经过DPO后模型似乎更喜欢说废话，把一段话说的很冗长，看起来很“客气”。另外，在格式上会更加喜欢分条，即使本来不需要，也更加喜欢markdown格式。

注：强化学习挺难训的，试过训练1，2，3，4个epoch，1个时候看起来变化不大，2个就是上面的效果，3个就成了复读机，4个就直接不输出了。

**NanoChat和qwen对比：**

在一些开放式问题，或者事实性问题上问，看起来是不会出错。但是NanoChat在数学问题上是灾难性的，1+1时能算出来的，再复杂一点就开始胡编乱造了，即使关了采样取概率最高的词也是如此，可能是样本中数学相关的样本太少了。反观qwen，铿锵有力，直接给出了答案hhh

------

**参考链接 & 感谢一下优秀的论文或项目**

- https://github.com/jingyaogong/minimind
- https://github.com/DLLXW/baby-llama2-chinese
- https://github.com/charent/Phi2-mini-Chinese
- https://github.com/AI-Study-Han/Zero-Chatgpt
- https://github.com/wangru8080/LLM_Trainer
