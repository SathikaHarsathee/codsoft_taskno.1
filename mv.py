import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('punkt')

traindata_path = "train_data.txt"
train_data = pd.read_csv(traindata_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

testdata_path = "test_data.txt"
test_data = pd.read_csv(testdata_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')

plt.figure(figsize=(16, 8))
sns.countplot(data=train_data, y='Genre', order=train_data['Genre'].value_counts().index, palette='rainbow', hue='Genre', legend=False)
plt.xlabel('COUNT', fontsize=15, fontweight='bold')
plt.ylabel('GENRE', fontsize=15, fontweight='bold')
plt.show()

plt.figure(figsize=(16, 10))
sns.scatterplot(data=train_data, y='Genre', x=train_data.groupby('Genre').cumcount(), palette='rainbow', marker='o', hue='Genre', c='orange')
plt.xlabel('COUNT', fontsize=14, fontweight='bold')
plt.ylabel('GENRE', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(18, 9))
counts = train_data['Genre'].value_counts()
sns.barplot(x=counts.index, y=counts, palette='spring_r', hue=counts.index, legend=False)
plt.xlabel('GENRE', fontsize=14, fontweight='bold')
plt.ylabel('COUNTS', fontsize=14, fontweight='bold')
plt.title('Distribution of Genres', fontsize=16, fontweight='bold')
plt.xticks(rotation=90, fontsize=14, fontweight='bold')
plt.show()

stemmer = LancasterStemmer()
stopwords_set = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')
    text = re.sub(r'http\S+', '', text)

    text = re.sub(r"[^a-zA-Z+']", ' ', text)
    text = re.sub(r'pic.\S+', '', text)

    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stopwords_set and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()

    return text

train_data['Text_cleaning'] = train_data['Description'].apply(clean_text)
test_data['Text_cleaning'] = test_data['Description'].apply(clean_text)

train_data['length_Text_cleaning'] = train_data['Text_cleaning'].apply(len)

plt.figure(figsize=(10, 10))
sns.histplot(data=train_data, x='length_Text_cleaning', bins=20, kde=True, color='hotpink')
plt.xlabel('LENGTH', fontsize=15, fontweight='bold')
plt.ylabel('FREQUENCY', fontsize=15, fontweight='bold')
plt.title('DISTRIBUTION_OF_LENGTHS', fontsize=15, fontweight='bold')
plt.show()

tf_idf_vectorizer = TfidfVectorizer()

X_train_data = tf_idf_vectorizer.fit_transform(train_data['Text_cleaning'])
X_test = tf_idf_vectorizer.transform(test_data['Text_cleaning'])

X = X_train_data
y = train_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred_val = classifier.predict(X_val)

accuracy_val = accuracy_score(y_val, y_pred_val)
print("Accuracy of Validation:", accuracy_val)
print(classification_report(y_val, y_pred_val, zero_division=1))

X_test_predictions = classifier.predict(X_test)
test_data['Predicted_Genre'] = X_test_predictions

test_data.to_csv('predicted_genres.csv', index=False)

print(test_data)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(train_data['Description']))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Cleaned Text', fontsize=16, fontweight='bold')
plt.show()
