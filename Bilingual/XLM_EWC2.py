import torch
from transformers import BertTokenizer
import pandas as pd
import csv
import os
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_path = './English_2/'
data_path2 = './Weibo_2/'

df = pd.read_csv(os.path.join(data_path, "train.tsv"), delimiter='\t')
df_dev = pd.read_csv(os.path.join(data_path, "dev.tsv"), delimiter='\t')
df_test = pd.read_csv(os.path.join(data_path, "test.tsv"), delimiter='\t')

df2 = pd.read_csv(os.path.join(data_path2, "train.tsv"), delimiter='\t')

MAX_LEN = 128

model_path = "./roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 提取语句并处理
# 训练集部分
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def token(tokenizer, df, df_dev, df_test):
    sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df.txt.values]
    labels = df.label.values
    # 这里0表示不积极,1表示积极
    labels = list(map(lambda x: 0 if x == 0 else 1, [x for x in labels]))
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
    # 将分割后的句子转化成数字  word-->idx
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    # 做PADDING，大于128做截断，小于128做PADDING
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # 建立mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 验证集部分
    dev_sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df_dev.txt.values]
    dev_labels = df_dev.label.values
    dev_labels = list(map(lambda x: 0 if x == 0 else 1, [x for x in dev_labels]))
    dev_tokenized_sents = [tokenizer.tokenize(sent) for sent in dev_sentencses]
    dev_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in dev_tokenized_sents]
    dev_input_ids = pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    dev_attention_masks = []
    for seq in dev_input_ids:
        dev_seq_mask = [float(i > 0) for i in seq]
        dev_attention_masks.append(dev_seq_mask)

    # 测试集部分
    test_sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df_test.txt.values]
    test_labels = df_test.label.values
    test_labels = list(map(lambda x: 0 if x == 0 else 1, [x for x in test_labels]))
    test_tokenized_sents = [tokenizer.tokenize(sent) for sent in test_sentencses]
    test_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in test_tokenized_sents]
    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    test_attention_masks = []
    for seq in test_input_ids:
        test_seq_mask = [float(i > 0) for i in seq]
        test_attention_masks.append(test_seq_mask)

    # 构建训练集、验证集、测试集的dataloader
    train_inputs = torch.tensor(input_ids)
    validation_inputs = torch.tensor(dev_input_ids)
    test_inputs = torch.tensor(test_input_ids)

    train_labels = torch.tensor(labels)
    validation_labels = torch.tensor(dev_labels)
    test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(attention_masks)
    validation_masks = torch.tensor(dev_attention_masks)
    test_masks = torch.tensor(test_attention_masks)

    batch_size = 8

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader


def tokentrain(tokenizer, df):
    # 构建验证集
    sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df.txt.values]
    labels = df.label.values

    labels = list(map(lambda x: 0 if x == 0 else 1, [x for x in labels]))
    # dev_labels=[to_categorical(i, num_classes=3) for i in dev_labels]
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    masks = torch.tensor(attention_masks)
    batch_size = 8
    # Create the DataLoader for our validation set.
    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


train_dataloader, validation_dataloader, test_dataloader = token(tokenizer, df, df_dev, df_test)
train_dataloader2 = tokentrain(tokenizer, df2)

from transformers import BertForSequenceClassification, AdamW, BertConfig

# 装载微调模型
model = XLMRobertaForSequenceClassification.from_pretrained("./XLM_C_result")
model.to(device)
# print(model.cuda())    # changed by yuemei

# 定义优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5)

# 学习率预热，训练时先从小的学习率开始训练
from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


# 计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 输出格式化时间
import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


from tqdm import trange
import torch.nn.functional as F

print("begin English counting")
# 计算重要度矩阵
params = {n: p for n, p in model.named_parameters() if p.requires_grad}  # 模型的所有参数
# p.requires_grad = true 训练时才会修改这层参数，model.named_parameters() 获取模型所有参数
# n：name p：param
_means = {}  # 初始化要把参数限制在的参数域
for n, p in params.items():
    _means[n] = p.clone().detach()

precision_matrices = {}  # 重要度
for n, p in params.items():
    precision_matrices[n] = p.clone().detach().fill_(0)  # 取zeros_like

model.eval()
for step, batch in enumerate(train_dataloader2):
    model.zero_grad()
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    ############ 核心代码 #############
    loss = F.nll_loss(F.log_softmax(output[0], dim=1), b_labels)
    # 计算labels对应的(正确分类的)对数概率，并把它作为loss func衡量参数重要度
    loss.backward()  # 反向传播计算导数
    for n, p in model.named_parameters():
        precision_matrices[n].data += p.grad.data ** 2 / len(train_dataloader2)
        ########### 计算对数概率的导数，然后反向传播计算梯度，以梯度的平方作为重要度 ########

print("importance counting complete!")


def trainer(train_dataloader, validation_dataloader, test_dataloader, model, optimizer, scheduler, precision_matrices,
            weight_decay_ewc):
    # 训练部分
    train_loss_set = []
    epochs = 3
    for _ in trange(epochs, desc="Epoch"):
        t0 = time.time()
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            batch = tuple(t.to(device) for t in batch)  # 将数据放置在GPU上
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            # print("loss:",loss)

            ### 核心代码 ###
            # 额外计算EWC的L2 loss
            ewc_loss = 0
            for n, p in model.named_parameters():
                _loss = precision_matrices[n] * (p - _means[n]) ** 2
                ewc_loss += _loss.sum()
            loss += weight_decay_ewc * ewc_loss

            train_loss_set.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("ewc_loss: {}".format(ewc_loss))
        print("Chinese Train loss: {}".format(tr_loss / nb_tr_steps))
        print("Chinese  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        t0 = time.time()
        # 验证集
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        print("Chinese Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        print("Chinese Validation took: {:}".format(format_time(time.time() - t0)))

    print("Chinese Training complete!")

    # 在测试集上进行测试
    t0 = time.time()
    model.eval()

    # Tracking variables
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    predictions, true_labels = [], []

    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
        # Calculate the accuracy for this batch of test sentences.
        tmp_test_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        test_accuracy += tmp_test_accuracy

        # Track the number of batches
        nb_test_steps += 1
    print("Chinese Test Accuracy: {0:.4f}".format(test_accuracy / nb_test_steps))
    print("Chinese Test took: {:}".format(format_time(time.time() - t0)))

    # f1值
    f1_set = []

    print('Calculating for each batch...')

    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        f1 = f1_score(true_labels[i], pred_labels_i)
        f1_set.append(f1)
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the F1
    F1 = f1_score(flat_true_labels, flat_predictions)
    print('f1-score: %.3f' % F1)

    count = 0
    for i in range(len(flat_true_labels)):
        if int(flat_predictions[i]) == int(flat_true_labels[i]):
            count += 1
    print("正确率: {0:.3f}".format(count / len(flat_true_labels)))

    output_dir2 = './XLM_CE_EWC_1e-5_result/'
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir2)

    return test_accuracy / nb_test_steps


# 装载预训练bert模型
weight_decay_ewc = 1e-5
result = trainer(train_dataloader, validation_dataloader, test_dataloader, model, optimizer, scheduler,
                 precision_matrices, weight_decay_ewc)

data_path2 = './Weibo_2/'
df_test2 = pd.read_csv(os.path.join(data_path2, "test.tsv"), delimiter='\t')


def tokentest(tokenizer, df):
    sentencses = ['[CLS] ' + sent + ' [SEP]' for sent in df.txt.values]
    labels = df.label.values

    labels = list(map(lambda x: 0 if x == 0 else 1, [x for x in labels]))
    # dev_labels=[to_categorical(i, num_classes=3) for i in dev_labels]
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    masks = torch.tensor(attention_masks)
    batch_size = 8
    # Create the DataLoader for our validation set.
    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


test_dataloader2 = tokentest(tokenizer, df_test2)

# test
t0 = time.time()
model.eval()
# Tracking variables
test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0
predictions, true_labels = [], []

# Evaluate data for one epoch
for batch in test_dataloader2:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
    # Get the "logits" output by the model. The "logits" are the output
    # values prior to applying an activation function like the softmax.
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)
    # Calculate the accuracy for this batch of test sentences.
    tmp_test_accuracy = flat_accuracy(logits, label_ids)

    # Accumulate the total accuracy.
    test_accuracy += tmp_test_accuracy

    # Track the number of batches
    nb_test_steps += 1

print("EWC English Test Accuracy: {0:.4f}".format(test_accuracy / nb_test_steps))
print("EWC English  Test took: {:}".format(format_time(time.time() - t0)))

# f1值
f1_set = []
# Evaluate each test batch using f1_score
print('Calculating for each batch...')

for i in range(len(true_labels)):
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    f1 = f1_score(true_labels[i], pred_labels_i)
    f1_set.append(f1)
# Combine the predictions for each batch into a single list of 0s and 1s.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]
# Calculate the F1
F1 = f1_score(flat_true_labels, flat_predictions)
print('f1-score: %.3f' % F1)

count = 0
for i in range(len(flat_true_labels)):
    if int(flat_predictions[i]) == int(flat_true_labels[i]):
        count += 1
print("正确率: {0:.3f}".format(count / len(flat_true_labels)))







