# -*- coding: utf-8 -*-
"""Object_oriented_structure.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1paEfxIyhJ6Vt9koe_e0VCstpUXMf_Dqj
"""


import pandas as pd
from transformers import BertModel
from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
import nltk
import jsonlines
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class DataType:
    string_type = ['title', 'abstract']
    float_type = ['num_low_freq_words', 'numbers_in_Abs', 'pub_year']
    boolean_value = ['has_mark', 'has_acronym', 'has_colon', 'gender']

class Input2Model:
    def __init__(self, df):
        self.df = df
        self.feature_input = []
        self.abstract_max_length = 128
        self.title_max_length = 128
        # self.vanue_max_length = 16

    def text_preprocessing(text):
        # Remove '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)

        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)

        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_year(self): 
        self.df['have_published'] = (2021 - self.df['year'] + 1)

    def preprocess_abstract(self, abstract):
        # remove github link
        new_abstract = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', abstract)
        return new_abstract

    def model_input(self, common_word_set):
        for index, row in self.df.iterrows():
            abstract_features = Abstract2Features(row[3])
            numbers_in_Abs = abstract_features.num_features()
          
            title_features = Title2Features(row[2])
            has_acronym, has_mark, has_colon = title_features.feature_style()
            # subarea = title_features.feature_topic()
            low_freq_words = title_features.feature_novelty(common_word_set)
            model_feature_input = {
                'has_acronym': has_acronym, # bool
                'has_mark': has_mark,  # bool
                'has_colon': has_colon, # bool
                'num_low_freq_words': low_freq_words, # float
                'numbers_in_Abs': numbers_in_Abs # float
            }
            self.feature_input.append(model_feature_input)

            
        meta_info_df = self.df[['title', 'abstract', 'citationCount', 'year', 'have_published']]
        feature_input_df = pd.DataFrame(self.feature_input)
        meta_info_df = meta_info_df.reset_index(drop=True)
        model_input_df = pd.concat([meta_info_df, feature_input_df], axis=1)
        train_df, val_df = train_test_split(model_input_df, test_size=0.1, random_state=2020)
        test_df, val_df = train_test_split(val_df, test_size=0.5, random_state=2020)
        return train_df, val_df, test_df

    def preprocess_for_BERT(self, given_df):
        title_input_ids = []
        title_attention_masks = []
        abstract_input_ids = []
        abstract_attention_masks = []
        # subarea_input_ids = []
        # subarea_attention_masks = []
        title = given_df.title.values
        abstract = given_df.abstract.values
        # subarea = given_df.subarea.values
        # VenueBinarizer = preprocessing.LabelBinarizer().fit(given_df["venue"])

        # one-hot encode for venue
        
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        for i in range(len(title)):
            encoded_title = tokenizer.encode_plus(
                text=title[i],    # Preprocess sentence
                add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
                max_length=self.title_max_length, # Max length to truncate/pad
                pad_to_max_length=True,           # Pad sentence to max length
                return_attention_mask=True        # Return attention mask
            )
            title_input_ids.append(encoded_title.get('input_ids'))
            title_attention_masks.append(encoded_title.get('attention_mask'))

            encoded_abstract = tokenizer.encode_plus(
                text=self.preprocess_abstract(abstract[i]),    # Preprocess sentence
                add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
                max_length=self.abstract_max_length, # Max length to truncate/pad
                pad_to_max_length=True,           # Pad sentence to max length
                return_attention_mask=True        # Return attention mask
            )
            abstract_input_ids.append(encoded_abstract.get('input_ids'))
            abstract_attention_masks.append(encoded_abstract.get('attention_mask'))

            # encoded_subarea = tokenizer.encode_plus(
            #     text=subarea[i],    # Preprocess sentence
            #     add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
            #     max_length=self.subarea_max_length, # Max length to truncate/pad
            #     pad_to_max_length=True,           # Pad sentence to max length
            #     return_attention_mask=True        # Return attention mask
            # )
            # subarea_input_ids.append(encoded_subarea.get('input_ids'))
            # subarea_attention_masks.append(encoded_subarea.get('attention_mask'))

        # encoded_venue = VenueBinarizer.transform(given_df["venue"])
        return title_input_ids, title_attention_masks, abstract_input_ids, abstract_attention_masks

class Citation:
    def __init__(self, raw_citation, num_years):

        # this_year = datetime.xxxx  # 2021
        # num_years = this_year - publication_year + 1
        self.raw_citation = raw_citation
        self.num_years = num_years
        self.log_citation = np.log(raw_citation)
        average_citation = self.average_citation()
        return average_citation
    
    def average_citation(self):
        return self.raw_citation / self.num_years

class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, title_input_ids, title_attention_masks, abstract_input_ids, abstract_attention_masks, df):
        self.title_input_ids = title_input_ids
        self.title_attention_masks = title_attention_masks
        self.abstract_input_ids = abstract_input_ids
        self.abstract_attention_masks = abstract_attention_masks
        # self.subarea_input_ids = subarea_input_ids
        # self.subarea_attention_masks = subarea_attention_masks
        self.has_acronym = df.has_acronym.values
        self.has_colon = df.has_colon.values
        self.has_mark = df.has_mark.values 
        self.num_low_freq_words = df.num_low_freq_words.values
        self.numbers_in_Abs = df.numbers_in_Abs.values
        self.year = df.year.values
        self.citation = df.citationCount.values
        self.have_published = df.have_published.values
        # self.encoded_venue = encoded_venue
    
    def __len__(self):
        return len(self.title_input_ids)
    
    def __getitem__(self, index):
        title_ids = np.asarray(self.title_input_ids[index]).reshape(-1,1)
        title_attn = np.asarray(self.title_attention_masks[index]).reshape(-1,1)
        title = (title_ids, title_attn)

        abstract_ids = np.asarray(self.abstract_input_ids[index]).reshape(-1,1)
        abstract_attn = np.asarray(self.abstract_attention_masks[index]).reshape(-1,1)
        abstract = (abstract_ids, abstract_attn)

        # subarea_ids = np.asarray(self.subarea_input_ids[index]).reshape(-1,1)
        # subarea_attn = np.asarray(self.subarea_attention_masks[index]).reshape(-1,1)
        # subarea = (subarea_ids, subarea_attn)
        
        # y = Citation(self.citation[index], self.have_published[index])

        return {'title': title, 'abstract':abstract, 'year': self.year[index], 'has_acronym': self.has_acronym[index], 'has_colon': self.has_colon[index], 'has_mark': self.has_mark[index], 'num_low_freq_words': self.num_low_freq_words[index],
                'numbers_in_Abs': self.numbers_in_Abs[index], 'citation': self.citation[index]}

class BertClassifier(nn.Module):
    def __init__(self, selected_keys, output_dimension, freeze_bert=False):

        super(BertClassifier, self).__init__()
        self.selected_keys = selected_keys
        self.datatype = DataType()
        num_float = 0
        num_bool = 0
        num_string = 0
        for key in selected_keys:
            if key in self.datatype.float_type:

                num_float += 1

            elif key in self.datatype.string_type:
                num_string += 1
            else:
                num_bool += 1
    
        float_hidden, float_out = 8, 16
        bool_out_size = 8
        self.string2emb = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=True)
        self.float2emb = nn.Sequential(
            nn.Linear(1, float_hidden),
            nn.ReLU(),
            nn.Linear(float_hidden, float_out),
            nn.ReLU(),
        )
        self.bool2emb = nn.Linear(num_bool, bool_out_size)
        output_size = 768 * num_string + float_out * num_float
        if num_bool != 0:
            output_size += 8
        h1, h2, h3 = 512, 256, 64
        self.classifier = nn.Sequential(
            nn.Linear(output_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(h3, output_dimension)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        # layers

    def forward(self, batch):
        '''
        each batch is
        {(input_ids, attention_mask)
            'tweet': [[input_ids1, input_ids2 ...], [attention_mask1, attention_mask2, ...]], # string
            'abstract': [abstract1, abstract2, abstract3, ...], # string
            'text_feature1': [True, False, True],
            'text_feature2': [True, False, True],
            'text_feature3': [True, False, True],
            'publication_year': float, # every int should be returned to float
        }
        '''
        bool_concat = []
        first = True
        
        output = torch.empty((2,2))
        # print(batch)
        for key, value in batch.items():
            if key in self.selected_keys:
                # print(key)
                if key in self.datatype.float_type:
                    value = value.float().to(device)
                    float_emb = self.float2emb(value.reshape(-1,1))
                    if first:
                        output = float_emb
                        first = False
                    else:
                        output = torch.cat((output, float_emb),1)
                elif key in self.datatype.string_type:
                    # print("enter string")
                    string_emb = self.string2emb(input_ids=value[0][:,:,0].to(device), attention_mask=value[1][:,:,0].to(device))
                    last_hidden_state_cls = string_emb[0][:, 0, :]
                    if first:
                        # print("enter first")
                        output = last_hidden_state_cls
                        first = False
                    else:
                        # print("enter second")
                        output = torch.cat((output, last_hidden_state_cls),1)
                else:
                    bool_concat.append(value.float())
    
        if bool_concat:
            num_bool = len(bool_concat)
            batch_size = len(bool_concat[0])
            bool_array = torch.empty([num_bool,batch_size])
            # print(bool_array.size())
            for i in range(num_bool):
                # temp = np.array(bool_concat[i])
                  bool_array[i] = torch.from_numpy(np.array(bool_concat[i]))
            # bool_concat = np.array(bool_concat).astype(int)
            # bool_concat = np.array(bool_concat)
            # print(bool_concat)
            # bool_concat = torch.from_numpy(np.array(bool_concat))
            bool_array = bool_array.to(device)
            # print(bool_array)
            # print(torch.transpose(bool_array,0,1))
            bool_emb = self.bool2emb(torch.transpose(bool_array,0,1))
            output = output = torch.cat((output, bool_emb),1)
        labels = batch['citation'].to(device)
        logits = self.classifier(output)
        return logits, labels

class Abstract2Features:
    def __init__(self, abstract):
        self.abstract = abstract
        self.tokens = word_tokenize(abstract)

    def num_features(self):
        numbers = [i for i in self.tokens if i.replace('.', '', 1).isdigit()]
        return len(numbers)

class Title2Features:
    def __init__(self, title):
        self.title = title
        self.tokens = word_tokenize(title)

    def feature_style(self):
        tokens = self.tokens
        has_acronym = False
        for i in tokens:
            upper = 0
            for c in i:
                if c.isupper():
                    upper += 1
            if upper >= 2:
                has_acronym = True
                break
        if ":" in tokens:
            if tokens.index(':') == 1:
                has_acronym = True
        # has_acronym = any(len([c.isupper() for c in i]) >= 2 for i in tokens) or (tokens.index(':') == 1)
        pattern_mark = '[\?\!]'
        has_mark = re.search(pattern_mark, self.title)!=None
        has_colon = True if any(i == ':' for i in tokens) else False
        # has_acronym: you check whether (1) there are words which has more than 2 capital letters, e.g., GloVe, BERT, OpinionFinder
        # OR (2) it is in the format of "Babytalk: Understanding and generating simple image descriptions"
        
        return has_acronym, has_mark, has_colon

    # def feature_topic(self):
    #     # pre_calculated_features is an instance of FeaturesByNLPModels()
    #     subareas = FeaturesByNLPModels(self.title)
    #     return subareas.title2subarea

    def feature_novelty(self, common_word_set):
        # common_word_set is what you pre-calculate on all paper title+abstract (tokenized version) of words larger than 100 occurrences
        # import nltk
        # new_text = nltk.word_tokenize(text)
        # We should avoid double tokenization: new_new_text = nltk.word_tokenize(new_text)

        low_freq_words = {i for i in self.tokens if i not in common_word_set}
        num_low_freq_words = len(low_freq_words)
        return num_low_freq_words # a number that shows how many low-frequency words are in the title.

class CommonSense:
    nlp_subareas = ['Computational Social Science and Cultural Analytics',
                    'Dialogue and Interactive Systems',
                    'Discourse and Pragmatics',
                    'Ethics and NLP',
                    'Generation',
                    'Information Extraction',
                    'Information Retrieval and Text Mining',
                    'Interpretability and Analysis of Models for NLP',
                    'Language Grounding to Vision, Robotics and Beyond',
                    'Linguistic Theories, Cognitive Modeling, and Psycholinguistics',
                    'Machine Learning for NLP',
                    'Machine Translation and Multilinguality',
                    'NLP Applications',
                    'Phonology, Morphology, and Word Segmentation',
                    'Question Answering',
                    'Resources and Evaluation',
                    'Semantics: Lexical',
                    'Semantics: Sentence-level Semantics, Textual Inference, and Other Areas',
                    'Sentiment Analysis, Stylistic Analysis, and Argument Mining',
                    'Speech and Multimodality',
                    'Summarization',
                    'Syntax: Tagging, Chunking and Parsing',
                    ]  # source: https://www.2022.aclweb.org/callpapers

# class FeaturesByNLPModels:
#     def __init__(self, title):
#         self.title2subarea = self.get_subarea(title)

#     def get_subarea(self, title):
#         # list_of_titles should be in its raw form, no tokenization, no lowercase
#         # reference: https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681
#         from transformers import pipeline
#         classifier = pipeline("zero-shot-classification")
#         # classifier = pipeline("zero-shot-classification", device=0) # to utilize GPU
#         candidate_labels = CommonSense().nlp_subareas

#         results = classifier(title, candidate_labels)
#         subareas = results["labels"][0]
#         return subareas

def initialize_model(selected_keys, output_dimension, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(selected_keys, output_dimension, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=1e-5,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      weight_decay=1e-3
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=1, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            # b_input_ids, b_attn_mask, a_input_ids, a_attn_mask, train_meta, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits, labels = model(batch)

            # Compute loss and accumulate the loss values
            # labels = batch['num_of_likes'].to(device)
            loss = loss_fn(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        # b_input_ids, b_attn_mask, a_input_ids, a_attn_mask, val_meta, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits, labels = model(batch)

        # labels = batch['num_of_likes'].to(device)
        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_json('new.jsonl', lines=True)
df.drop_duplicates(subset ="title", keep = False, inplace = True)


df = df[pd.notnull(df['abstract'])]
df = df[pd.notnull(df['title'])]
df['citationCount'] = (df['citationCount'] >= 6).astype(int)

from sklearn.feature_extraction.text import CountVectorizer

content_set = []
for index, row in df.iterrows():
    content_set.append(row[2])
    content_set.append(row[3])
content_set = filter(None, content_set)
cv = CountVectorizer()
word_count_vec = cv.fit_transform(content_set).toarray()
word_name = cv.get_feature_names()
freq = np.sum(word_count_vec, axis=0)
common_word_set = []
for index, item in enumerate(word_name):
    if freq[index] > 1000:
        common_word_set.append(word_name[index])


nltk.download('punkt')
input2model = Input2Model(df)
input2model.preprocess_year()
train_df, val_df, test_df = input2model.model_input(common_word_set)
train_df.to_pickle("train_df.pkl")
val_df.to_pickle("val_df.pkl")
test_df.to_pickle("test_df.pkl")
train_title_input_ids, train_title_attention_masks, train_abstract_ids, train_abstract_attention_masks= input2model.preprocess_for_BERT(train_df)
val_title_input_ids, val_title_attention_masks, val_abstract_ids, val_abstract_attention_masks = input2model.preprocess_for_BERT(val_df)
test_title_input_ids, test_title_attention_masks, test_abstract_ids, test_abstract_attention_masks = input2model.preprocess_for_BERT(test_df)

# from sklearn import preprocessing
# # train_df = train_df.apply(preprocessing.LabelEncoder().fit_transform)
# # train_df

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# # target_col = 'num_of_likes'
# # x = train_df[['follower_count', 'friends_count', 'retweet_count', 'has_link', 'num_emoji', 'num_tag', 'num_at', 'has_num', 'has_mark']]
# # y = train_df[target_col]
# # # y = train_df.iloc[:, train_df.columns == target_col]

# # select = SelectKBest(score_func=chi2, k=5)
# # z = select.fit_transform(x,y)
# # print(z)

batch_size = 32

train_data = PaperDataset(train_title_input_ids, train_title_attention_masks, train_abstract_ids, train_abstract_attention_masks, train_df)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

val_data = PaperDataset(val_title_input_ids, val_title_attention_masks, val_abstract_ids, val_abstract_attention_masks, val_df)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=True)

test_data = PaperDataset(test_title_input_ids, test_title_attention_masks, test_abstract_ids, test_abstract_attention_masks, test_df)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)

selected_keys = ['abstract']

loss_fn = nn.CrossEntropyLoss()
set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(selected_keys, output_dimension = 2)
train(bert_classifier, train_dataloader, val_dataloader, epochs=6, evaluation=True)