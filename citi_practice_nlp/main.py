import json
import random

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim # 嵌入维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.vocab_size = vocab_size # 词汇大小
        self.tag_to_ix = tag_to_ix # 标签转为下标
        self.tagset_size = len(tag_to_ix) # 目标取值范围大小

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim) # 嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag过渡参数矩阵。条目i，j是从*j过渡到*i*的分数。
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function 用正演算法计算配分函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score. START_TAG拥有所有分数。
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop 包装在一个变量中，这样我们将获得自动backprop
        forward_var = init_alphas

        # Iterate through the sentence 重复这个句子
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep 此时的前向张量
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag 广播发射分数：无论前一个标签如何，它都是相同的
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i trans_score的第i个条目是从i转换到next_tag的得分
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    #序列预测
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
#极大取相反数作为loss
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#用于设置随机初始化的种子，即上述的编号，编号固定，每次获取的随机数固定

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if shuffle == True:
        random.shuffle(full_list)
    sublist1 = full_list[:offset]
    sublist2 = full_list[offset:]
    return sublist2, sublist1   # 训练集 测试集

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 6  # 由于标签一共有NAME NOTIONAL 和 TICKER WORD START STOP 6个，所以embedding_dim为6
HIDDEN_DIM = 4  # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2

# Make up some training data 每一个句子对应的词性都有正确的标签
# 1.1 读取文件，修改标签
with open('./data/data.json', 'r') as obj:
    pre_data = json.load(obj)

training_data = []
for sentence_dict in pre_data:
    text = sentence_dict['text'].split()
    tag = np.repeat('WORD', len(text)).tolist()
    # 利用标签下标更新新的tag
    labels = sentence_dict['label']
    for label in labels:
        posl = label[1][0]
        posr = label[1][1]
        temp_str = sentence_dict['text'][posl:posr].split()
        if len(temp_str) >1:
            str_index = text.index(temp_str[0])
            str_index_r = text.index(temp_str[-1])
            for i in range(str_index, str_index_r+1):
                tag[i] = label[0]
        else:
            str_index = text.index(temp_str[0])
            tag[str_index] = label[0]
    data = (text, tag)
    training_data.append(data)

training_data, testing_data = data_split(training_data, ratio=0.3, shuffle=False)   # 不打乱顺序
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

#处理数据集中句子的词，不重复的将句子中的词拿出来并标号
#设置一个word_to_ix存储句子中每一个单词
#拿出每一个句子跟其对应的标签，用sentence与tag循环每一个句子
#比如第0次抽出来的就是第一个句子"the wall …money"，与他的标签"B I I …0"。
#现在只存储出现过的单词(不管标签)

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            #word_to ix没有word这句单词时候，就存储进去
            #并且给每一个新的word赋值，赋的值就是长度。
            word_to_ix[word] = len(word_to_ix)
for sentence, tags in testing_data:
    for word in sentence:
        if word not in word_to_ix:
            #word_to ix没有word这句单词时候，就存储进去
            #并且给每一个新的word赋值，赋的值就是长度。
            word_to_ix[word] = len(word_to_ix)

# 保存词库
with open('ciku.json', 'w') as f:
    json.dump(word_to_ix, f)

#将5个标签存到tag_to_ix的字典中
tag_to_ix = {"WORD": 0, "NAME": 1, "NOTIONAL": 2, "TICKER": 3, START_TAG: 4, STOP_TAG: 5}

#将句子输入到BILSTM-CRF模型
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10):  # again, normally you would NOT do 300 epochs, it is toy data
    loss_vec = []
    if epoch % 5 == 0:
        for sentence, tags in testing_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss_vec.append(loss.item())
        print('loss of epoch ' + str(epoch) + ' is ' + str(sum(loss_vec) / len(loss_vec)))
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()#反向求导
        optimizer.step()#参数更新

torch.save(model.state_dict(), 'model_parameter.pkl')

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!