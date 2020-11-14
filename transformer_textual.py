import json
import glob
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm, tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import normalize, scale
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE

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

def train_and_test():
    # prepare data
    fileNameList = glob.glob('C:/YYQ/PGproject/PreProcessing/processed_features_facenet/*.pkl')
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

    # exit()
    print(text_list)
    lens = [len(a.split()) for a in text_list]
    print(min(lens), max(lens))
    exit()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pg = tokenizer.batch_encode_plus(text_list, max_length=128, pad_to_max_length=True)
    '''print(len(pg))
    for k in pg.keys():
        print(k, len(pg[k]), [len(kk) for kk in pg[k]])'''

    x = pg['input_ids']
    token_type_ids = pg['token_type_ids']
    attention_mask = pg['attention_mask']
    '''for xx in x:
        print(xx)'''

    x, token_type_ids, attention_mask = np.array(x), np.array(token_type_ids), np.array(attention_mask)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=5)
    cv5_ids = list(skf.split(x, labels))

    sp = cv5_ids[0]
    train_l, train_labels = x[sp[0]], labels[sp[0]]
    # train_data, train_labels = sm.fit_sample(train_data, train_labels)
    test_l, test_labels = x[sp[1]], labels[sp[1]]
    print(train_l.shape)

    train_token_type_ids, test_token_type_ids, train_attention_mask, test_attention_mask = token_type_ids[sp[0]], \
                                           token_type_ids[sp[1]], attention_mask[sp[0]], attention_mask[sp[1]]

    # shuffle training data for batch reading
    n_train = len(train_l)
    n_eval = len(test_l)
    perm = np.random.permutation(n_train)
    train_l = train_l[perm]
    train_labels = np.array(train_labels)[perm]
    train_token_type_ids, train_attention_mask = train_token_type_ids[perm], train_attention_mask[perm]

    train_l, test_l, train_labels, test_labels, train_token_type_ids, test_token_type_ids = torch.LongTensor(train_l), \
                                                                            torch.LongTensor(test_l), \
                                                                            torch.LongTensor(train_labels), \
                                                                            torch.LongTensor(test_labels), \
                                                                            torch.LongTensor(train_token_type_ids), \
                                                                            torch.LongTensor(test_token_type_ids)

    train_attention_mask, test_attention_mask = torch.FloatTensor(train_attention_mask), \
                                                torch.FloatTensor(test_attention_mask)

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to('cuda')
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
    model = BertForSequenceClassification(config).to('cuda')
    # print(model(train_l[:32], token_type_ids=train_token_type_ids[:32], attention_mask=train_attention_mask[:32], labels=train_labels[:32])[1])

    eval_every = 5
    batch_size = 32
    test_batch_size = 8
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

    for ep in range(max_epochs):
        idx = 0
        avg_loss = 0
        n_batch = 0
        model.train()
        while idx < n_train:
            optimizer.zero_grad()
            batch_l = train_l[idx:(idx + batch_size)].to('cuda')
            batch_ty = train_token_type_ids[idx:(idx + batch_size)].to('cuda')
            batch_am = train_attention_mask[idx:(idx + batch_size)].to('cuda')
            ans = train_labels[idx:(idx + batch_size)].to('cuda')
            idx += batch_size
            preds = model(input_ids=batch_l, token_type_ids=batch_ty, attention_mask=batch_am, labels=ans)
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
                test_batch_l = test_l[idx:(idx + test_batch_size)].to('cuda')
                test_batch_ty = test_token_type_ids[idx:(idx + test_batch_size)].to('cuda')
                test_batch_am = test_attention_mask[idx:(idx + test_batch_size)].to('cuda')
                test_ans = test_labels[idx:(idx + test_batch_size)].to('cuda')
                # time.sleep(20)
                # exit()
                test_pred = model(input_ids=test_batch_l,
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

            '''scores = model(train_data, train_lens)
            _, train_preds = scores.data.cpu().max(1)
            print("training set: %f" % (float(sum(train_preds.numpy() == train_labels.cpu().numpy())) / len(train_preds.numpy())))
            print(eval_preds.numpy())'''
            print(float(sum(eval_preds == test_labels.cpu().numpy())) / len(eval_preds))
            print(precison, recall, fscore, support)


if __name__ == "__main__":
    train_and_test()

