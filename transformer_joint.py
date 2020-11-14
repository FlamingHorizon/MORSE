import json
import glob
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm, tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, scale
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE
import re
import os

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold


def get_optimizers(model, learning_rate, adam_epsilon, weight_decay, num_training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


def align_tokenized_features(tokenizer, text_list, visual_features, audio_features):
    max_seq_len = max([len(tokenizer.tokenize(t)) for t in text_list])
    n_samples = len(text_list)
    new_visual = np.zeros((n_samples, max_seq_len, visual_features[0].shape[-1]))
    new_audio = np.zeros((n_samples, max_seq_len, audio_features[0].shape[-1]))
    n = 0 # iterator of samples
    for t in text_list:
        # try rebuild original sentence with space spliting from the tokenized list
        # t_len = len(t.split())
        i = 0 # iterator of tokens, new_visual and new_audio
        j = 0 # iterator of visual_features, audio_features
        tk = tokenizer.tokenize(t)
        old_len = len(visual_features[n])
        check = []
        while i < len(tk):
            if j >= old_len:
                i_end = i
                while i_end < len(tk):
                    new_visual[n][i_end][:] += visual_features[n][j - 1][:]
                    new_audio[n][i_end][:] += audio_features[n][j - 1][:]
                    i_end += 1
                break
            check.append((tk[i], text_list[n].split()[j]))
            if tk[i] in ['\'', ':']:
                if tk[i-1] == 'so' and tk[i+1] == 'on' or tk[i-1] == 'up' and tk[i+1] == 'to':
                    new_visual[n][i][:] += visual_features[n][j-1][:]
                    new_audio[n][i][:] += audio_features[n][j-1][:]
                    i += 1
                else:
                    new_visual[n][i][:] += visual_features[n][j-1][:]
                    new_visual[n][i+1][:] += visual_features[n][j-1][:]
                    new_audio[n][i][:] += audio_features[n][j - 1][:]
                    new_audio[n][i + 1][:] += audio_features[n][j - 1][:]
                    i += 2
            elif tk[i] == '-':
                if i == 0 or tk[i+1] in ['and','again','because'] or (tk[i-1] == 'oil' and tk[i+1] == 'oil') or (tk[i-1] == 'oil' and tk[i+1] == 'like'):
                    new_visual[n][i][:] = visual_features[n][j][:]
                    new_audio[n][i][:] = audio_features[n][j][:]
                    i += 1
                    j += 1
                else:
                    new_visual[n][i][:] += visual_features[n][j - 1][:]
                    new_visual[n][i + 1][:] += visual_features[n][j - 1][:]
                    new_audio[n][i][:] += audio_features[n][j - 1][:]
                    new_audio[n][i + 1][:] += audio_features[n][j - 1][:]
                    i += 2
            elif tk[i] == '/':
                if re.match('[0-9]+', tk[i-1]) or tk[i-1] in ['medium', 'afternoon', 'either']:
                    new_visual[n][i][:] += visual_features[n][j - 1][:]
                    new_visual[n][i + 1][:] += visual_features[n][j - 1][:]
                    new_audio[n][i][:] += audio_features[n][j - 1][:]
                    new_audio[n][i + 1][:] += audio_features[n][j - 1][:]
                    i += 2
                else:
                    new_visual[n][i][:] = visual_features[n][j][:]
                    new_audio[n][i][:] = audio_features[n][j][:]
                    i += 1
                    j += 1
            elif tk[i] == '+':
                if tk[i+1] in ['anti', 'fields']:
                    new_visual[n][i][:] = visual_features[n][j][:]
                    new_audio[n][i][:] = audio_features[n][j][:]
                    i += 1
                    j += 1
                else:
                    new_visual[n][i][:] += visual_features[n][j - 1][:]
                    new_audio[n][i][:] += audio_features[n][j - 1][:]
                    i += 1
            elif tk[i][0] == '#' or tk[i] == '%':
                new_visual[n][i][:] += visual_features[n][j - 1][:]
                new_audio[n][i][:] += audio_features[n][j - 1][:]
                i += 1
            elif tk[i] == '$':
                new_visual[n][i][:] += visual_features[n][j][:]
                new_visual[n][i+1][:] += visual_features[n][j][:]
                new_audio[n][i][:] += audio_features[n][j][:]
                new_audio[n][i + 1][:] += audio_features[n][j][:]
                i += 2
                j += 1
            else:
                new_visual[n][i][:] = visual_features[n][j][:]
                new_audio[n][i][:] = audio_features[n][j][:]
                i += 1
                j += 1

        n += 1
        # print(j, old_len)


    return new_visual, new_audio


def seq2array(list_in, max_len):
    n_samples = len(list_in)
    feature_dim = list_in[0].shape[-1]
    print(feature_dim)
    out_feat = np.zeros((n_samples, max_len, feature_dim))
    token_ids = np.zeros((n_samples, max_len))
    attention_mask = np.zeros((n_samples, max_len))
    for i in range(n_samples):
        for j in range(min(max_len, len(list_in[i]))):
            out_feat[i][j][:] += list_in[i][j][:]
            attention_mask[i][j] = 1
    # print(out_feat)
    return out_feat, token_ids, attention_mask


def select_balanced_id(ids, labels):
    labels_used = labels[ids]
    neg, neu, pos = [], [], []
    for i in range(len(labels_used)):
        if labels_used[i] == 0:
            neg.append(ids[i])
        elif labels_used[i] == 1:
            neu.append(ids[i])
        else:
            pos.append(ids[i])
    neg, neu, pos = np.array(neg), np.array(neu), np.array(pos)
    print(len(neg), len(neu), len(pos))
    n_sample = len(neg)
    perm = np.random.permutation(len(neg))
    neg = neg[perm]
    perm = np.random.permutation(len(neu))
    neu = neu[perm][:n_sample]
    perm = np.random.permutation(len(pos))
    pos = pos[perm][:n_sample]
    concat_all = np.concatenate([neg,neu,pos], axis=-1)
    perm = np.random.permutation(len(concat_all))
    concat_all = concat_all[perm]
    print(concat_all, len(concat_all))
    return concat_all




class jointModalBert(nn.Module):
    def __init__(self,
                 config,
                 dim_emb=768):
        super(jointModalBert, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.visual_proj = nn.Linear(config.visual_dim + config.hidden_size, config.hidden_size)
        self.audio_proj = nn.Linear(config.audio_dim + config.hidden_size, config.hidden_size)
        self.joint_proj = nn.Linear(config.audio_dim + config.visual_dim + config.hidden_size, config.hidden_size)
        self.seqBert = BertForSequenceClassification(config)

    def forward(self, input_ids=None, input_visual=None, input_audio=None, token_type_ids=None, attention_mask=None, labels=None):
        word_embeds = self.word_embeddings(input_ids)
        if input_visual is None and input_audio is not None:
            pre_embs = torch.cat((word_embeds, input_audio), dim=-1)
            inputs_embeds = self.audio_proj(pre_embs)
        elif input_audio is None and input_visual is not None:
            pre_embs = torch.cat((word_embeds, input_visual), dim=-1)
            inputs_embeds = self.visual_proj(pre_embs)
        elif input_audio is not None and input_visual is not None:
            pre_embs = torch.cat((word_embeds, input_visual, input_audio), dim=-1)
            # print(word_embeds.size(), input_visual.size(), input_audio.size(), pre_embs.size())
            inputs_embeds = self.joint_proj(pre_embs)
        else:
            inputs_embeds = word_embeds
        outputs = self.seqBert(attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels,
                               inputs_embeds=inputs_embeds)
        return outputs

def train_and_test():
    # prepare data
    fileNameList = glob.glob('processed_features_facenet/*.pkl')
    # print(fileNameList)
    # basic features
    # text-list and tf-idf
    text_list = []
    labels = []
    visual_features = []
    audio_features = []
    for file_name in fileNameList:
        data_point = pkl.load(open(file_name, 'rb'))
        clip_name, label, transcription, smoothed_seq = data_point[0], data_point[1], data_point[2], data_point[3]
        # print(label, transcription)
        # continue
        labels.append(label)
        text_list.append(transcription)
        # average visual features
        # visual_seq = np.stack([w['landmark_feature'] for w in smoothed_seq], axis=0)
        visual_seq = np.stack([w['facenet_feature'].squeeze() for w in smoothed_seq], axis=0)
        # visual_seq = scale(visual_seq)
        # print(visual_seq)
        # visual_seq = visual_seq - np.mean(visual_seq, axis=0)
        # print(visual_seq.shape)
        # visual_mean = np.mean(visual_seq, axis=0)
        visual_features.append(visual_seq)
        # average audio features
        audio_seq = np.stack([w['audio_grp'] for w in smoothed_seq], axis=0)
        # audio_seq = scale(audio_seq)
        # print(audio_seq.shape)
        # audio_mean = np.mean(audio_seq, axis=0)
        audio_features.append(audio_seq)

    # print(text_list)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    visual_features, audio_features = align_tokenized_features(tokenizer, text_list, visual_features, audio_features)

    # print(visual_features[0], audio_features[0])
    pg = tokenizer.batch_encode_plus(text_list, max_length=128, pad_to_max_length=True)

    x = pg['input_ids']
    token_type_ids = pg['token_type_ids']
    attention_mask = pg['attention_mask']

    x, token_type_ids, attention_mask = np.array(x), np.array(token_type_ids), np.array(attention_mask)

    max_seq_len = 128
    visual_features = visual_features[:,:max_seq_len,:]
    audio_features = audio_features[:,:max_seq_len,:]
    # visual_features, token_type_ids, attention_mask = seq2array(visual_features, max_seq_len)
    # audio_features, _, _ = seq2array(audio_features, max_seq_len)
    visual_dim = visual_features.shape[-1]
    audio_dim = audio_features.shape[-1]
    print(visual_dim, audio_dim)


    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=5)
    cv5_ids = list(skf.split(token_type_ids, labels))

    sp = cv5_ids[0]
    selected_sp = sp[0]
    balanced_ids = select_balanced_id(selected_sp, labels)
    train_l, train_labels = x[sp[0]], labels[sp[0]]
    train_v = visual_features[sp[0]]
    train_a = audio_features[sp[0]]
    '''train_l, train_labels = x[balanced_ids], labels[balanced_ids]
    train_v = visual_features[balanced_ids]
    train_a = audio_features[balanced_ids]'''
    # train_data, train_labels = sm.fit_sample(train_data, train_labels)

    test_l, test_labels = x[sp[1]], labels[sp[1]]
    test_v = visual_features[sp[1]]
    test_a = audio_features[sp[1]]
    print(train_v.shape)

    train_token_type_ids, test_token_type_ids, train_attention_mask, test_attention_mask = token_type_ids[sp[0]], \
                                           token_type_ids[sp[1]], attention_mask[sp[0]], attention_mask[sp[1]]

    '''train_token_type_ids, test_token_type_ids, train_attention_mask, test_attention_mask = token_type_ids[balanced_ids], \
                                                                                           token_type_ids[sp[1]], \
                                                                                           attention_mask[balanced_ids], \
                                                                                           attention_mask[sp[1]]'''

    # shuffle training data for batch reading
    n_train = len(train_v)
    n_eval = len(test_v)
    perm = np.random.permutation(n_train)
    train_l, train_a, train_v = train_l[perm], train_a[perm], train_v[perm]
    print(train_l.shape, train_a.shape, train_v.shape)
    train_labels = np.array(train_labels)[perm]
    train_token_type_ids, train_attention_mask = train_token_type_ids[perm], train_attention_mask[perm]

    train_l, test_l, train_labels, test_labels, train_token_type_ids, test_token_type_ids = torch.LongTensor(train_l), \
                                                                            torch.LongTensor(test_l), \
                                                                            torch.LongTensor(train_labels), \
                                                                            torch.LongTensor(test_labels), \
                                                                            torch.LongTensor(train_token_type_ids), \
                                                                            torch.LongTensor(test_token_type_ids)

    train_a, test_a, train_v, test_v = torch.FloatTensor(train_a), torch.FloatTensor(test_a), \
                                       torch.FloatTensor(train_v), torch.FloatTensor(test_v)

    train_attention_mask, test_attention_mask = torch.FloatTensor(train_attention_mask), \
                                                torch.FloatTensor(test_attention_mask)

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to('cuda')
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
    config.visual_dim = visual_dim
    config.audio_dim = audio_dim
    # model = BertForSequenceClassification(config).to('cuda')
    model = jointModalBert(config).to('cuda')
    # print(model(train_l[:32], token_type_ids=train_token_type_ids[:32], attention_mask=train_attention_mask[:32], labels=train_labels[:32])[1])

    eval_every = 5
    batch_size = 32
    test_batch_size = 4
    max_epochs = 500
    t_total = math.ceil(n_train / batch_size) * max_epochs
    lr = 2e-5
    epsilon = 1e-8
    max_grad_norm = 1.0
    weight_decay = 0.0

    optimizer, scheduler = get_optimizers(model, learning_rate=lr, adam_epsilon=epsilon, weight_decay=weight_decay,
                                          num_training_steps=t_total)

    # loss_fn = torch.nn.CrossEntropyLoss().cuda()
    model.train()
    model.zero_grad()

    day = time.localtime().tm_mday
    minute = time.localtime().tm_min
    hour = time.localtime().tm_hour
    save_dir = 'fine_tuning_checkpoints/' + '-%d-%d-%d/' %(day, hour, minute)
    # os.mkdir(save_dir)

    for ep in range(max_epochs):
        idx = 0
        avg_loss = 0
        n_batch = 0
        model.train()
        while idx < n_train:
            optimizer.zero_grad()
            batch_l = train_l[idx:(idx + batch_size)].to('cuda')
            batch_v = train_v[idx:(idx + batch_size)].to('cuda')
            batch_a = train_a[idx:(idx + batch_size)].to('cuda')
            batch_ty = train_token_type_ids[idx:(idx + batch_size)].to('cuda')
            batch_am = train_attention_mask[idx:(idx + batch_size)].to('cuda')
            ans = train_labels[idx:(idx + batch_size)].to('cuda')
            idx += batch_size
            preds = model(input_ids=batch_l, input_visual=batch_v, input_audio=batch_a, token_type_ids=batch_ty, attention_mask=batch_am, labels=ans)
            loss = preds[0]
            # print(preds, ans)
            loss.backward()
            # print(loss.data.cpu().numpy())
            avg_loss += loss.data.cpu().numpy()
            n_batch += 1.

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_loss = avg_loss / n_batch
        print("epoch: %d avg_loss: %f" % (ep + 1, avg_loss))

        del batch_l, batch_ty, batch_am
        torch.cuda.empty_cache()
        # time.sleep(20)

        if ep % eval_every == 0:
            idx = 0
            model.eval()
            eval_preds = np.array([])
            while idx < n_eval:
                test_batch_v = test_v[idx:(idx + test_batch_size)].to('cuda')
                test_batch_l = test_l[idx:(idx + test_batch_size)].to('cuda')
                test_batch_a = test_a[idx:(idx + test_batch_size)].to('cuda')
                test_batch_ty = test_token_type_ids[idx:(idx + test_batch_size)].to('cuda')
                test_batch_am = test_attention_mask[idx:(idx + test_batch_size)].to('cuda')
                test_ans = test_labels[idx:(idx + test_batch_size)].to('cuda')
                # time.sleep(20)
                # exit()
                test_pred = model(input_ids=test_batch_l,
                                  input_visual=test_batch_v,
                                  input_audio=test_batch_a,
                                  token_type_ids=test_batch_ty,
                                  attention_mask=test_batch_am,
                                  labels=test_ans)
                scores = test_pred[1]
                _, batch_eval_preds = scores.data.cpu().max(1)
                eval_preds = np.concatenate((eval_preds, batch_eval_preds), axis=-1)
                idx += test_batch_size
            # metrics
            precison, recall, fscore, support = precision_recall_fscore_support(test_labels.cpu().numpy(), eval_preds,
                                                                                labels=[0, 1, 2], average=None)

            print(float(sum(eval_preds == test_labels.cpu().numpy())) / len(eval_preds))
            print(precison, recall, fscore, support)
            print('saving:')

            model_dir = save_dir + '%d' % (ep+1)
            # os.mkdir(model_dir)
            # model.save_pretrained(model_dir)



if __name__ == "__main__":
    train_and_test()
