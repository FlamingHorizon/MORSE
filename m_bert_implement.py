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

from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable

from transformers.modeling_utils import PreTrainedModel, prune_linear_layer



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


class DotAttention(nn.Module):
    def __init__(self, config):
        super(DotAttention, self).__init__()
        self.v_gate = nn.Linear(config.hidden_size + config.visual_dim, 1)
        self.a_gate = nn.Linear(config.hidden_size + config.audio_dim, 1)
        self.v_proj = nn.Linear(config.visual_dim, config.hidden_size)
        self.a_proj = nn.Linear(config.audio_dim, config.hidden_size)
        self.bias = Variable(torch.FloatTensor(0.001 * np.ones((config.hidden_size, 1)))).to('cuda')

    def forward(self, input_emb, input_audio, input_visual):
        cls_emb = input_emb[:,0,:]
        word_emb = input_emb[:,1:,:]
        input_visual = input_visual[:,1:,:]
        input_audio = input_audio[:, 1:, :]
        N, T, D = word_emb.size()
        g_v = nn.functional.relu(self.v_gate(torch.cat((word_emb, input_visual), dim=-1)))
        g_a = nn.functional.relu(self.a_gate(torch.cat((word_emb, input_audio), dim=-1)))
        v_hidden = self.v_proj(input_visual)
        a_hidden = self.a_proj(input_audio)
        # print(g_v.size(),g_a.size(), self.bias.unsqueeze(0).unsqueeze(1).squeeze(3).size())
        h_m = g_v.expand(N,T,D) * v_hidden + g_a.expand(N,T,D) * a_hidden + self.bias.unsqueeze(0).unsqueeze(1).squeeze(3).expand(N,T,D)
        ratio = word_emb.norm(dim=-1).unsqueeze(2).expand(N,T,D) / h_m.norm(dim=-1).unsqueeze(2).expand(N,T,D)
        alpha = torch.min(ratio * 1, torch.ones(N,T,D).to('cuda'))
        out_1 = word_emb + alpha * h_m
        # print(cls_emb.size(), out_1.size())
        out = torch.cat((cls_emb.unsqueeze(1), out_1), dim=1)
        return out


BERT_START_DOCSTRING = r"""
"""

BERT_INPUTS_DOCSTRING = r"""
"""

@add_start_docstrings(

    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",

    BERT_START_DOCSTRING,

)

class mBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.inject = DotAttention(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):

        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):

        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):

        """ Prunes heads of the model.

            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}

            See base class PreTrainedModel

        """

        for layer, heads in heads_to_prune.items():

            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        input_visual=None,
        input_audio=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        inserted_output = self.inject(input_emb=embedding_output, input_audio=input_audio, input_visual=input_visual)
        encoder_outputs = self.encoder(
            inserted_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[

            1:

        ]  # add hidden_states and attentions if they are here

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:

            if self.num_labels == 1:

                #  We are doing regression

                loss_fct = MSELoss()

                loss = loss_fct(logits.view(-1), labels.view(-1))

            else:

                loss_fct = CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def train_and_test():

    visual_features = pkl.load(open('tf_features/visual_features_facenet.pkl', 'rb'))
    audio_features = pkl.load(open('tf_features/audio_features.pkl', 'rb'))
    x = pkl.load(open('tf_features/linguistic_features.pkl', 'rb'))
    token_type_ids = pkl.load(open('tf_features/token_type_ids.pkl', 'rb'))
    attention_mask = pkl.load(open('tf_features/attention_mask.pkl', 'rb'))
    labels = pkl.load(open('tf_features/labels.pkl', 'rb'))
    cv5_ids = pkl.load(open('tf_features/cv5_ids.pkl', 'rb'))
    visual_dim = visual_features.shape[-1]
    audio_dim = audio_features.shape[-1]
    print(visual_dim, audio_dim)

    sp = cv5_ids[0]
    train_l, train_labels = x[sp[0]], labels[sp[0]]
    train_v = visual_features[sp[0]]
    train_a = audio_features[sp[0]]

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

    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
    config.visual_dim = visual_dim
    config.audio_dim = audio_dim
    bert_external = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    bert_insert = mBertModel(config)
    bert_insert.embeddings = bert_external.embeddings
    bert_insert.encoder = bert_external.encoder
    bert_insert.pooler = bert_external.pooler
    model = mBertModel(config).to('cuda')

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
            torch.cuda.empty_cache()

        avg_loss = avg_loss / n_batch
        print("epoch: %d avg_loss: %f" % (ep + 1, avg_loss))

        del batch_l, batch_v, batch_a, batch_ty, batch_am, ans
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
                torch.cuda.empty_cache()

            del test_batch_l, test_batch_v, test_batch_a, test_batch_ty, test_batch_am, test_ans
            torch.cuda.empty_cache()
            # metrics
            precison, recall, fscore, support = precision_recall_fscore_support(test_labels.cpu().numpy(), eval_preds,
                                                                                labels=[0, 1, 2], average=None)

            print(float(sum(eval_preds == test_labels.cpu().numpy())) / len(eval_preds))
            print(precison, recall, fscore, support)
            print('saving:')

            '''model_dir = save_dir + '%d' % (ep+1)
            os.mkdir(model_dir)
            model.save_pretrained(model_dir)'''



if __name__ == "__main__":
    train_and_test()