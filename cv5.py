import json
import glob
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
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


def choose_modality(linguistic_features, visual_features, audio_features, mod='all'):
    if mod == 'l':
        return np.concatenate([linguistic_features], axis=1)
    elif mod == 'a':
        return np.concatenate([audio_features], axis=1)
    elif mod == 'v':
        return np.concatenate([visual_features], axis=1)
    elif mod == 'la':
        return np.concatenate([linguistic_features, audio_features], axis=1)
    elif mod == 'lv':
        return np.concatenate([linguistic_features, visual_features], axis=1)
    elif mod == 'av':
        return np.concatenate([audio_features, visual_features], axis=1)
    elif mod == 'all':
        return np.concatenate([linguistic_features, visual_features, audio_features], axis=1)
    else:
        print('ha?')
        exit()


# load data
fileNameList = glob.glob('processed_features_facenet/*.pkl')
print(fileNameList)

text_list = []
labels = []
visual_features = []
audio_features = []
au_features = []
for file_name in fileNameList:
    data_point = pkl.load(open(file_name, 'rb'))
    clip_name, label, transcription, smoothed_seq = data_point[0], data_point[1], data_point[2], data_point[3]
    au_file = 'processed_features/' + file_name.split('\\')[1]
    au = pkl.load(open(au_file, 'rb'))

    _, _, _, smoothed_seq_au = au[0], au[1], au[2], au[3]

    labels.append(label)
    text_list.append(transcription)
    # average visual features
    au_seq = np.stack([w['landmark_feature'] for w in smoothed_seq_au], axis=0)
    visual_seq = np.stack([w['facenet_feature'] for w in smoothed_seq], axis=0)

    visual_mean = np.mean(visual_seq, axis=0)
    au_mean = np.mean(au_seq, axis=0)
    visual_features.append(visual_mean)
    au_features.append(au_mean)
    # average audio features
    audio_seq = np.stack([w['audio_grp'] for w in smoothed_seq], axis=0)

    audio_mean = np.mean(audio_seq, axis=0)
    audio_features.append(audio_mean)

# print(text_list)
vectorizer = TfidfVectorizer(min_df=5)
docMatrix = vectorizer.fit_transform(text_list)

linguistic_features = docMatrix.toarray()
linguistic_features = scale(linguistic_features)
# print(linguistic_features)

visual_features = np.array(visual_features).squeeze()
# visual_features = scale(visual_features)
# visual_features = normalize(visual_features)
au_features = scale(np.array(au_features).squeeze())
visual_features = np.concatenate((visual_features, au_features), axis=-1)
# visual_features = au_features
visual_features = normalize(visual_features)
visual_features = scale(visual_features)
# print(visual_features)
audio_features = np.array(audio_features)
# audio_features = scale(audio_features)
# print(audio_features)
labels = np.array(labels)

# linguistic_features = normalize(linguistic_features)

full_data = choose_modality(linguistic_features, visual_features, audio_features, mod='all')

print(full_data.shape)

# full_data = normalize(full_data)
# full_data = scale(full_data)

perm = np.random.permutation(len(full_data))
full_data = full_data[perm]
labels = np.array(labels)[perm]

# initialize cv5
skf = StratifiedKFold(n_splits=5)
cv5_ids = list(skf.split(full_data, labels))
# print(cv5_ids)

# initialize model
# lin_clf = svm.SVC(decision_function_shape='ovo', probability=True)
# lin_clf = svm.LinearSVC()
# lin_clf = LogisticRegression()
# lin_clf = svm.SVC(kernel='sigmoid')
# lin_clf = MLPClassifier((256,256), activation='relu', max_iter=1000)
# lin_clf = RandomForestClassifier(n_estimators=5000, max_depth=2, random_state=0)
single_clf = tree.DecisionTreeClassifier(max_depth=1)
# single_clf = LogisticRegression()
lin_clf = RUSBoostClassifier(base_estimator=single_clf, n_estimators=5000)

# initialize booster
sm = SMOTE(random_state=42)

# perform cv5
precision_avg = []
recall_avg = []
fscore_avg = []
acc_avg = 0.
for sp in cv5_ids:
    train_data, train_labels = full_data[sp[0]], labels[sp[0]]
    # train_data, train_labels = sm.fit_sample(train_data, train_labels)
    test_data, test_labels = full_data[sp[1]], labels[sp[1]]

    lin_clf.fit(train_data, train_labels)
    pred = lin_clf.predict(test_data)
    print(sp[1])
    print(pred)
    print(test_labels)
    # metrics
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, pred, labels=[0, 1, 2],
                                                                         average=None)
    acc = float(sum(pred == test_labels)) / len(test_labels)
    print(precision, recall, fscore, support, acc)
    precision_avg.append(precision)
    recall_avg.append(recall)
    fscore_avg.append(fscore)
    acc_avg += acc
precision, recall, fscore = np.mean(precision_avg, axis=0), np.mean(recall_avg, axis=0), np.mean(fscore_avg, axis=0)
acc_avg = acc_avg / len(cv5_ids)
print('cv5-avg:')
print(precision, recall, fscore, acc_avg)

