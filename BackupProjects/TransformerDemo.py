import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2  # 定义批次大小（batch size），这里设置为2，表示同时处理2个样本

# 单词表大小
max_num_src_words = 8  # 定义源句子词表的最大词汇数量（不包括padding），这里设置为8
max_num_tgt_words = 8  # 定义目标句子词表的最大词汇数量（不包括padding），这里设置为8

model_dim = 8  # 定义模型的嵌入维度（model dimension），即每个词向量的维度，这里设置为8

# token序列的最大长度
max_src_seq_len = 5  # 定义源序列的最大长度，这里设置为5，用于填充padding
max_tgt_seq_len = 5  # 定义目标序列的最大长度，这里设置为5，用于填充padding
max_position_len = 5  # 定义位置编码的最大位置长度，这里设置为5，与序列最大长度相关

src_len = torch.Tensor([2, 4]).to(torch.int32)  # 定义源序列的实际长度张量，这里手动设置为[2, 4]，表示第一个样本长度2，第二个长度4，并转换为int32类型
tgt_len = torch.Tensor([4, 3]).to(torch.int32)  # 定义目标序列的实际长度张量，这里手动设置为[4, 3]，表示第一个样本长度4，第二个长度3，并转换为int32类型

print(src_len)
print(tgt_len)

# 单词索引构成源句子和目标句子， 构建batch， 并且做了padding， 默认值为0
# 构建源序列（src_seq）：对于每个样本，根据其实际长度生成随机词索引（1到max_num_src_words），然后进行padding到最大长度，使用0填充右侧
src_seq = torch.cat(
    [ # list 在[]内使用for l in src_len
        torch
        .unsqueeze( #增加一个维度
            F.pad(
                torch.randint(
                    1, max_num_src_words, (L,)  # 最小为1 最大为最大的词典数，并且仅有1维，1维的大小为L
                ),
                (0, max(src_len) - L) # 对L长度意外的位置填0
            ), 0
        )
        for L in src_len #对词源长度的每个元素进行创建张量，超出长度的为0
    ]
)
# 构建目标序列（tgt_seq）：类似源序列，根据实际长度生成随机词索引（1到max_num_tgt_words），padding到最大长度，使用0填充右侧
tgt_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max(tgt_len) - L)), 0) for L in tgt_len])


# TODO 构造word embedding
# 创建源词嵌入表（embedding table）：一个嵌入层，将词索引映射到model_dim维向量，词表大小为max_num_src_words+1（包括padding的0）
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
# 创建目标词嵌入表：类似源嵌入表，词表大小为max_num_tgt_words+1
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)
# 获取源序列的词嵌入：将src_seq输入嵌入表，得到形状为[batch_size, max_src_seq_len, model_dim]的张量
src_embedding = src_embedding_table(src_seq)
# 获取目标序列的词嵌入：类似，得到形状为[batch_size, max_tgt_seq_len, model_dim]的张量
tgt_embedding = tgt_embedding_table(tgt_seq)