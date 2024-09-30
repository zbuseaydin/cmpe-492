import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# ---------------------------
# Step 1: Download NLTK Data
# ---------------------------
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------------------
# Step 2: Load the "text8" Corpus
# ---------------------------
print("Loading the 'text8' corpus...")
dataset = api.load('text8')  # Generator object
corpus = [sentence for sentence in dataset]  # List of tokenized sentences
print(f"Number of sentences in Text8: {len(corpus)}")

# ---------------------------
# Step 3: Define Preprocessing Function
# ---------------------------
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sentence):
    """
    Preprocesses a single sentence:
    - Converts to lowercase
    - Tokenizes into words
    - Removes stopwords and punctuation
    - Removes non-alphabetic tokens
    - Lemmatizes the words
    """
    # Convert list of words to a single string
    text = ' '.join(sentence)
    
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]
    
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# ---------------------------
# Step 4: Apply Preprocessing
# ---------------------------
print("Preprocessing the corpus...")
processed_corpus = [preprocess_sentence(sentence) for sentence in corpus]
print(f"Number of processed sentences: {len(processed_corpus)}")

# ---------------------------
# Step 5: Initialize and Train the Word2Vec Model
# ---------------------------
print("Training the Word2Vec model...")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(
    sentences=processed_corpus,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10
)
print("Word2Vec model training complete.")

# ---------------------------
# Step 6: Save the Model
# ---------------------------
model.save('text8_word2vec.model')
print("Model saved to 'text8_word2vec.model'.")

# ---------------------------
# Step 7: Load the Model
# ---------------------------
loaded_model = Word2Vec.load('text8_word2vec.model')
print("Model loaded successfully.")

# ---------------------------
# Step 8: Print Interesting Results
# ---------------------------

# a. Finding Similar Words
target_word = 'computer'
if target_word in loaded_model.wv:
    similar_words = loaded_model.wv.most_similar(target_word, topn=10)
    print(f"\nWords similar to '{target_word}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
else:
    print(f"'{target_word}' not found in the vocabulary.")

# b. Word Analogies
positive = ['woman', 'king']
negative = ['man']
missing_words = [word for word in positive + negative if word not in loaded_model.wv]
if not missing_words:
    analogy = loaded_model.wv.most_similar(positive=positive, negative=negative, topn=1)
    print(f"\nKing - Man + Woman = {analogy[0][0]} (Similarity: {analogy[0][1]:.4f})")
else:
    print(f"Words not in vocabulary: {missing_words}")

# c. Similarity Between Two Words
word1 = 'computer'
word2 = 'machine'
if word1 in loaded_model.wv and word2 in loaded_model.wv:
    similarity = loaded_model.wv.similarity(word1, word2)
    print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
else:
    missing = [word for word in [word1, word2] if word not in loaded_model.wv]
    print(f"Words not found in the vocabulary: {missing}")

# d. Finding Outliers
words = ['breakfast', 'lunch', 'dinner', 'computer']
outliers = loaded_model.wv.doesnt_match(words)
print(f"\nOutlier in the group {words}: {outliers}")

# ---------------------------
# Step 9: Optional - Visualizing Word Embeddings with t-SNE
# ---------------------------
print("\nVisualizing word embeddings with t-SNE...")

# Define a list of words to visualize
visualization_words = ['computer', 'computers', 'computerized', 'computing', 'technology',
                      'machine', 'machines', 'automation', 'data', 'algorithm',
                      'king', 'queen', 'prince', 'princess', 'monarch']

# Filter words present in the vocabulary
filtered_words = [word for word in visualization_words if word in loaded_model.wv]

# Get the corresponding word vectors
word_vectors = [loaded_model.wv[word] for word in filtered_words]

# Initialize and fit t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_2d = tsne.fit_transform(word_vectors)

# Plot the words
plt.figure(figsize=(12, 8))
for i, word in enumerate(filtered_words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.title('Word Embeddings Visualization with t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
