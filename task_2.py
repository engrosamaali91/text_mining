# Importing necessary libraries
from collections import Counter
import pandas as pd

# Loading the dataset
file_path = 'emerging.dev.conll'  

# Reading the file
with open(file_path, 'r') as file:
    data = file.readlines()

# Parsing the dataset
parsed_data = [line.strip().split('\t') for line in data if line.strip()]

# Convert parsed data into a pandas DataFrame for easy processing
df = pd.DataFrame(parsed_data, columns=["Token", "Entity"])

# Initialize lists to store sentences and labels
sentences = []
labels = []

current_sentence = []
current_labels = []

# Loop through the DataFrame to group tokens and labels into sentences
for index, row in df.iterrows():
    token = row['Token']
    label = row['Entity']
    
    # When punctuation marks indicate the end of a sentence
    if token in ['.', '!', '?']:
        # Add current token and label to the ongoing sentence
        current_sentence.append(token)
        current_labels.append(label)
        
        # Append the sentence and labels to their respective lists
        sentences.append(current_sentence)
        labels.append(current_labels)
        
        # Reset for the next sentence
        current_sentence = []
        current_labels = []
    else:
        # Continue collecting tokens and labels
        current_sentence.append(token)
        current_labels.append(label)

# Convert the results to a DataFrame for saving or further analysis
grouped_df = pd.DataFrame({'Words': sentences, 'Labels': labels})

# Save the DataFrame to a CSV file 
grouped_df.to_csv('grouped_sentences_labels.csv', index=False)

# Display the grouped sentences and labels
print(grouped_df.head())
