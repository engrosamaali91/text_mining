from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


# Loading the dataset
file_path = 'emerging.dev.conll'  


# Reading the file
with open(file_path, 'r') as file:
    data = file.readlines()
# Parsing the dataset
 
parsed_data = [line.strip().split('\t') for line in data if line.strip()]
#print(parsed_data)



# Convert parsed data into a pandas DataFrame for easy processing
df = pd.DataFrame(parsed_data, columns=["Token", "Entity"])

# Counting the number of entities for each entity class
# First, we filter out the 'O' label as it represents non-entities.
entity_df = df[df['Entity'] != 'O']
# print(entity_df)

# Now, count how many times each entity label appears.
entity_counts = entity_df['Entity'].value_counts()

# Display the total number of entities for each class
# print("Number of Entities per Class:")
# print(entity_counts)

# function to find the top 5 tokens for each entity class
def find_top_tokens(tokens):
    token_counts = Counter(tokens)
    return token_counts.most_common(5)

# Finding the top 5 words for each entity class
# Group by the 'Entity' column and then count the frequency of tokens for each class.

top_tokens_per_entity = entity_df.groupby('Entity')['Token'].apply(find_top_tokens)


# Display the top 5 words for each entity class
print("\nTop 5 Words per Entity Class:")
print(top_tokens_per_entity)


# Number of Entities (total count across all classes)
total_entities = entity_df['Entity'].count()

# Display total number of entities
print(f"\nTotal Number of Entities in the Dataset: {total_entities}")



#visualization
plt.figure(figsize=(8, 6))
entity_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Entities per Class')
plt.xlabel('Entity Class')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Bar plots for each entity class to visualize top 5 tokens
for entity, tokens in top_tokens_per_entity.items():
    tokens_list, counts = zip(*tokens)  # Separate tokens and counts
    plt.figure(figsize=(6, 4))
    plt.bar(tokens_list, counts, color='skyblue')
    plt.title(f'Top 5 Tokens for {entity}')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()