import pickle as pkl
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, AutoModelForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import math
import time
import torch.nn as nn
from iqi_svdd import SVDD
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

import scipy.io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize, scale
from scipy.cluster.vq import whiten
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from iqi_svdd_visualize import Visualization as draw


def load_data():
    # visual_features = pkl.load(open('tf_features/visual_features_facenet.pkl','rb'))
    # audio_features = pkl.load(open('tf_features/audio_features.pkl', 'rb'))
    visual_features, audio_features = None, None
    x = pkl.load(open('tf_features/linguistic_features.pkl', 'rb'))
    token_type_ids = pkl.load(open('tf_features/token_type_ids.pkl', 'rb'))
    attention_mask = pkl.load(open('tf_features/attention_mask.pkl', 'rb'))
    labels = pkl.load(open('tf_features/labels.pkl', 'rb'))
    cv5_ids = pkl.load(open('tf_features/cv5_ids.pkl', 'rb'))

    return visual_features, audio_features, x, token_type_ids, attention_mask, labels, cv5_ids

def get_pretrained_embedding(word_ids, token_type_ids, attention_mask):
    # model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    model = BertForSequenceClassification.from_pretrained('fine_tuning_checkpoints\\-16-10-55\\151').to('cuda')
    # model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english').to('cuda')
    n_samples = len(word_ids)
    batch_size = 32
    idx = 0
    word_ids, token_type_ids, attention_mask = torch.LongTensor(word_ids), torch.LongTensor(token_type_ids), \
                                               torch.FloatTensor(attention_mask)
    all_reps = []
    while idx < n_samples:
        batch_l = word_ids[idx:(idx + batch_size)].to('cuda')
        batch_ty = token_type_ids[idx:(idx + batch_size)].to('cuda')
        batch_am = attention_mask[idx:(idx + batch_size)].to('cuda')
        idx += batch_size
        # rep_seq = model.distilbert(input_ids=batch_l, token_type_ids=batch_ty, attention_mask=batch_am)[0].data.cpu().numpy()
        rep_seq = model.bert(input_ids=batch_l, token_type_ids=batch_ty, attention_mask=batch_am)[0].data.cpu().numpy()
        # rep_vector = np.mean(rep_seq, axis=1)
        # rep_vector = rep_seq[:,0,:]
        rep_vector = rep_seq[:, :, :]
        all_reps.append(rep_vector)
    all_reps = np.concatenate(all_reps,axis=0)
    return all_reps


_, _, x, token_type_ids, attention_mask, labels, cv5_ids = load_data()
all_reps = get_pretrained_embedding(x, token_type_ids, attention_mask)
pkl.dump(all_reps, open('reps/fine_tuned_sp4.pkl','wb'))