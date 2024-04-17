# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
from tqdm import tqdm
import pandas as pd
import random
import ast


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
df = pd.read_csv('/kaggle/input/assignment-1-nlp/train.csv') # loading training data
data = []
for index, row in tqdm(df.iterrows()):
    data.append(ast.literal_eval(row['tagged_sentence'])) # changing data-type of entries from 'str' to 'list'

# %%
df = pd.read_csv('/kaggle/input/assignment-1-nlp/test_small.csv') # loading test data
test_data = {} 
for index, row in tqdm(df.iterrows()):
    test_data[row['id']] = ast.literal_eval(row['untagged_sentence']) # changing data-type of entries from 'str' to 'list'

# %%
def display_data(sentence_index):
    '''
        Input : 'sentence_index' (int) -> index of a sentence in training data
        Output: None
    '''
    sentence = data[sentence_index]
    print("TOKEN -> TAG")
    print('...')
    for token, tag in sentence:
        print(token, '>', tag)
sentence_index = random.choice(range(len(data)))
display_data(sentence_index)

# %% [markdown]
# Dict naam badlooooo

# %%
from collections import defaultdict
dict_train_tokens=defaultdict(int)
dict_train_tags=defaultdict(int)
# len_of_data= len(data)
for sentence in data:
    for token, tag in sentence:
        dict_train_tokens[token] += 1
        dict_train_tags[tag] += 1

# %%
len_train_tags = len(dict_train_tags)
len_train_tokens = len(dict_train_tokens)
print(len_train_tags)

# %%
# Assuming dict_train_tags is a dictionary where keys are tags
tags_list = list(dict_train_tags.keys())  # Extract tags as a list from the dictionary

# Use enumerate for a cleaner and more Pythonic approach to create mappings
dict_index_to_tags = {index: tag for index, tag in enumerate(tags_list)}
dict_tags_to_index = {tag: index for index, tag in enumerate(tags_list)}

# %%
# Assuming dict_train_tokens is a dictionary where keys are tokens
tokens_list = list(dict_train_tokens.keys())  # Extract tokens as a list from the dictionary

# Use enumerate for a cleaner approach to create mappings
dict_index_to_tokens = {index: token for index, token in enumerate(tokens_list)}
dict_tokens_to_index = {token: index for index, token in enumerate(tokens_list)}

# %%
dict_tags_to_index

# %%
emission_prob_matrix = np.zeros((len_train_tags, len_train_tokens))

# Populate the emission probability matrix
for sentence in data:
    for token, tag in sentence:
        row_index = dict_tags_to_index[tag]
        column_index = dict_tokens_to_index[token]
        emission_prob_matrix[row_index, column_index] += 1

# Normalize the matrix to convert counts to probabilities
emission_prob_matrix /= emission_prob_matrix.sum(axis=1, keepdims=True)

# %%
emission_prob_matrix.shape

# %%
import numpy as np

# Reinitialize the transition probability matrix as per the original dimensions provided
transition_prob_matrix = np.zeros((len_train_tags + 1, len_train_tags + 1))

# Iterate over sentences in the data to populate the transition probability matrix
for sentence in data:
    sentence_length = len(sentence)
    for i in range(sentence_length - 1):
        current_tag = sentence[i][1]
        next_tag = sentence[i + 1][1]

        # Update transitions from start tag to the first tag of the sentence
        if i == 0:
            transition_prob_matrix[len_train_tags, dict_tags_to_index[current_tag]] += 1
        
        # Update transitions from the last tag to the end tag placeholder
        if i == sentence_length - 2:
            transition_prob_matrix[dict_tags_to_index[next_tag], len_train_tags] += 1
        
        # Update transitions between tags within the sentence
        transition_prob_matrix[dict_tags_to_index[current_tag], dict_tags_to_index[next_tag]] += 1

# Normalize the matrix to convert counts to probabilities
for i in range(len(transition_prob_matrix)):
    row_sum = transition_prob_matrix[i, :].sum()
    if row_sum > 0:
        transition_prob_matrix[i, :] /= row_sum

print(f"Transition Probability Matrix Shape: {transition_prob_matrix.shape}")

# %%
print(transition_prob_matrix)

# %%
# Assuming 'df' is a pandas DataFrame with columns 'untagged_sentence' and 'id'
test_data_dict = df.set_index('id')['untagged_sentence'].to_dict()

# %%
dict_train_tokens

# %%
temp=np.array([[1,2,3],
              [4,5,6],
            [4,5,6]])
print(temp[0:-1,0])

# %%
def viterbi_algorithm(sentence, transition_matrix, emission_matrix):
    num_tags = len_train_tags  # Assuming tags_dict is defined globally
    sentence_length = len(sentence)
    probability_matrix = np.zeros((num_tags, sentence_length))
    backpointer_matrix = np.zeros((num_tags, sentence_length), dtype=int)
    
    # Initialize all tags list for reconstructing the path
    all_tags = []
    
    for position, word in enumerate(sentence):
        for tag_index in range(num_tags):
            if word not in dict_train_tokens:  # Handle unknown words
                emission_probability = 1
            else:
                emission_probability = emission_matrix[tag_index, dict_tokens_to_index[word]]
            
            if position == 0:
                probability_matrix[tag_index, position] = transition_matrix[num_tags, tag_index] * emission_probability
            else:
                transition_probabilities = transition_matrix[0:-1, tag_index] * probability_matrix[:, position - 1]
                best_previous_tag = np.argmax(transition_probabilities)
                probability_matrix[tag_index, position] = transition_probabilities[best_previous_tag] * emission_probability
                backpointer_matrix[tag_index, position] = best_previous_tag
                
    # Trace back the best path
    last_tag = np.argmax(probability_matrix[:, -1])
    all_tags.append(last_tag)
    
    for i in range(sentence_length - 1, 0, -1):
        all_tags.insert(0, backpointer_matrix[all_tags[0], i])
    
    # Create the tagged sentence using the best path
    tagged_sentence = [(word, dict_index_to_tags[tag]) for word, tag in zip(sentence, all_tags)]
    
    return tagged_sentence

# %%
def run_viterbi_on_test_data(limit, test_sentences_dict):
    processed_sentences = {}
    
    # Convert sentences in the test data dictionary to their tagged versions
    for sentence_id, sentence_str in test_sentences_dict.items():
        # Parsing the sentence string into a list (if necessary)
        sentence = ast.literal_eval(sentence_str) if isinstance(sentence_str, str) else sentence_str
        
        # Run the Viterbi algorithm on the parsed sentence
        tagged_sentence = viterbi_algorithm(sentence, transition_prob_matrix, emission_prob_matrix)
        
        # Store the result using the original sentence id
        processed_sentences[sentence_id] = tagged_sentence
        
        # Optionally limit the number of processed sentences
#         if limit and len(processed_sentences) >= limit:
#             break

    return processed_sentences

# %%
test_predictions = run_viterbi_on_test_data(5,test_data_dict)

# %%
predictions_df = pd.DataFrame(test_predictions,columns=['id', 'tagged_sentence'])

# Specify your desired CSV file path
output_csv_path = 'sha2_predictions_hmm.csv'

# Save the DataFrame to a CSV file
predictions_df.to_csv(output_csv_path, index=False)

# %%
import csv


csv_file_path = 'sha2_predictions_hmm.csv'


with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['id', 'tagged_sentence'])

    for key, values in test_predictions.items():
        writer.writerow([key, values])


csv_file_path

# %%
# cell to show the frequency of each distinct (slack or native) present in the training data
from collections import Counter
distinct_tags = []
word_tags = []
def store_tags():
    
    global distinct_tags
    global word_tags
    
    for sent in data:
        word_tags.append(('START','START'))
        for words, tag in sent:
            word_tags.extend([(tag, words)])
        word_tags.append(('END','END'))
    
store_tags()
tags=[]
for tag, words in word_tags:
    tags.append(tag)
distinct_tags=list(set(tags))
count_tags = {}
for tag, count in Counter(tags).items():
    count_tags[tag] = count

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.xticks(rotation='vertical')
plt.bar(range(len(count_tags)), list(count_tags.values()), align='center')
plt.xticks(range(len(count_tags)), list(count_tags.keys()))
plt.xlabel('Tag')
plt.ylabel('Count')
plt.show()

# %%
submission = {'id': [], 'tagged_sentence' : []} # dictionary to store tag predictions
# NOTE ---> ensure that tagged_sentence's corresponing 'id' is same as 'id' of corresponding 'untagged_sentence' in training data
def store_submission(sent_id, tagged_sentence):
    
    global submission
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)
    
def clear_submission():
    global submission
    submission = {'id': [], 'tagged_sentence' : []}

# %%
def hmm_tagger_util(sent_id, untagged_sentence):
    store_submission(sent_id, tagged_sentence)

# %%
def memm_tagger_util(sent_id, untagged_sentence):
    store_submission(sent_id, tagged_sentence)

# %%
# cell to implement tagger that allots random tags to words in a sentence

def random_tagger_util(sent_id, untagged_sentence):
    if(sent_id in list(submission['id'])):
        return
    tagged_sentence = []
    for word in untagged_sentence:
        tagged_sentence.append((word, random.choice(distinct_tags)))
    store_submission(sent_id, tagged_sentence)
    

# %%
for sent_id in tqdm(list(test_data.keys())):
    sent = test_data[sent_id]
    random_tagger_util(sent_id, sent)

# %%
path_to_directory = '/kaggle/working/'
pd.DataFrame(submission).to_csv(path_to_directory +' sample_submission.csv', index = False)


