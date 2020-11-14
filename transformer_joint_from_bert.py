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
import re
import os

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
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
    # optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


class jointModalBert(nn.Module):
    def __init__(self,
                 config,
                 dim_emb=768):
        super(jointModalBert, self).__init__()
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.visual_proj = nn.Linear(config.visual_dim + config.hidden_size, config.hidden_size)
        self.audio_proj = nn.Linear(config.audio_dim + config.hidden_size, config.hidden_size)
        self.joint_proj = nn.Linear(config.audio_dim + config.visual_dim + config.hidden_size, config.hidden_size)
        self.seqBert = BertForSequenceClassification(config)
        self.jointLayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_reps=None, input_visual=None, input_audio=None, token_type_ids=None, attention_mask=None, labels=None):
        word_embeds = input_reps
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
        inputs_embeds = self.jointLayerNorm(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds)
        outputs = self.seqBert(attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels,
                               inputs_embeds=inputs_embeds)
        return outputs

def train_and_test():

    visual_features = pkl.load(open('tf_features/visual_features_facenet.pkl', 'rb'))
    audio_features = pkl.load(open('tf_features/audio_features.pkl', 'rb'))
    x = pkl.load(open('reps/fine_tuned_sp4.pkl', 'rb'))
    token_type_ids = pkl.load(open('tf_features/token_type_ids.pkl', 'rb'))
    attention_mask = pkl.load(open('tf_features/attention_mask.pkl', 'rb'))
    labels = pkl.load(open('tf_features/labels.pkl', 'rb'))
    cv5_ids = pkl.load(open('tf_features/cv5_ids.pkl', 'rb'))
    visual_dim = visual_features.shape[-1]
    audio_dim = audio_features.shape[-1]
    print(visual_dim, audio_dim)

    sp = cv5_ids[4]
    train_l, train_labels = x[sp[0]], labels[sp[0]]
    train_v = visual_features[sp[0]]
    train_a = audio_features[sp[0]]
    # train_data, train_labels = sm.fit_sample(train_data, train_labels)

    test_l, test_labels = x[sp[1]], labels[sp[1]]
    test_v = visual_features[sp[1]]
    test_a = audio_features[sp[1]]
    print(train_v.shape)

    train_token_type_ids, test_token_type_ids, train_attention_mask, test_attention_mask = token_type_ids[sp[0]], \
                                           token_type_ids[sp[1]], attention_mask[sp[0]], attention_mask[sp[1]]

    # shuffle training data for batch reading
    n_train = len(train_v)
    n_eval = len(test_v)
    perm = np.random.permutation(n_train)
    train_l, train_a, train_v = train_l[perm], train_a[perm], train_v[perm]
    print(train_l.shape, train_a.shape, train_v.shape)
    train_labels = np.array(train_labels)[perm]
    train_token_type_ids, train_attention_mask = train_token_type_ids[perm], train_attention_mask[perm]

    train_l, test_l, train_labels, test_labels, train_token_type_ids, test_token_type_ids = torch.FloatTensor(train_l), \
                                                                            torch.FloatTensor(test_l), \
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
            preds = model(input_reps=batch_l, input_visual=batch_v, input_audio=batch_a, token_type_ids=batch_ty, attention_mask=batch_am, labels=ans)
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

            torch.cuda.empty_cache()

        del batch_l, batch_v, batch_a, batch_ty, batch_am, ans
        torch.cuda.empty_cache()
        avg_loss = avg_loss / n_batch
        print("epoch: %d avg_loss: %f" % (ep + 1, avg_loss))


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
                test_pred = model(input_reps=test_batch_l,
                                  input_visual=test_batch_v,
                                  input_audio=test_batch_a,
                                  token_type_ids=test_batch_ty,
                                  attention_mask=test_batch_am,
                                  labels=test_ans)
                scores = test_pred[1]
                _, batch_eval_preds = scores.data.cpu().max(1)
                eval_preds = np.concatenate((eval_preds, batch_eval_preds), axis=-1)
                idx += test_batch_size
                torch.cuda.empty_cache()

            del test_batch_l, test_batch_v, test_batch_a, test_batch_ty, test_batch_am, test_ans
            torch.cuda.empty_cache()
            # metrics
            precison, recall, fscore, support = precision_recall_fscore_support(test_labels.cpu().numpy(), eval_preds,
                                                                                labels=[0, 1, 2], average=None)

            print(float(sum(eval_preds == test_labels.cpu().numpy())) / len(eval_preds))
            print(precison, recall, fscore, support)
            # print('saving:')

            '''model_dir = save_dir + '%d' % (ep+1)
            os.mkdir(model_dir)
            model.save_pretrained(model_dir)'''



if __name__ == "__main__":
    train_and_test()
