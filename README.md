# MORSE
For paper "MORSE: MultimOdal sentiment analysis for Real-life SEttings"

## Data Format

According to privacy terms, we only release the pre-processed facenet and action unit features for visual modality.
We only release the extracted covarep feature sequences for acoustic modality.

All the transcripts are available.

## Loading MORSE

Dropbox [link](https://www.dropbox.com/s/yz0qohdp5hpdxaq/Dropbox_MORSE.zip?dl=0)

Features for Transformer models can be directly loaded from tf_features/;

The transcripts and visual/acoustic feature sequences aligned to each word can be found in processed_features/ (for action units), and processed_features_facenet/ (for facenet). They are organized in tuples (clip_name, label, transcription, smoothed_seq), where smoothed_seq has the form \[{'word','facenet_feature/landmark_feature','audio_grp'}\].

## Scripts
**cv5.py**:

Preliminary features and baselines with 5-fold cross validation

**m_bert_implement.py**: 

Our reproduction of the "Shifting Gate" joint fine-tuning method
(see arXiv preprint arXiv:1908.05787)

**get_textual_reps.py**:

Load the saved Transformer weights from step-1: language fine-tuning, compute and dump the language embeddings for step-2

**transformer_joint.py**:

The transformer-all model, trained from scratch

**transformer_textual.py**:

Transformer using linguistic modality only;
to train from scratch, use 
```python
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
model = BertForSequenceClassification(config).to('cuda')
```
**transformer_joint_from_bert.py**:

Read the language vectors from step-1 and run step-2 joint fine-tuning

**Pipeline**:
1.

Initialize a bert-pre-trained model using
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to('cuda')
```
in any of the scripts, train on our dataset, and save the checkpoints
2.

Use get_textual_reps.py to compute textual embedding with the checkpoint you select.
3.

Use transformer_joint_from_bert.py to load the textual embeddings (under reps/) and run joint fine-tuning for several epochs.

You can load reps/fine_tuned_sp4.pkl for quick results, it is the textual embeddings computed using a checkpoint in the step-1 fine-tuning, under cv5_ids\[4\] splitting.

## Dependencies

Python3

Pytorch

Transformers


