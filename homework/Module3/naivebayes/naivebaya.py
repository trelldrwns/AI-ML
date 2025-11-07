# A script to classify SMS messages as spam or ham using Naive Bayes.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/mnt/khome/requiem/Documents/programs/AI&ML/homework/Module3/naivebayes/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label_numeric'] = df['label'].map({'ham': 0, 'spam': 1})

print("--- Task 2: Bayes Theorem Analysis for the word 'free' ---")

total_messages = len(df)
spam_count = len(df[df['label'] == 'spam'])
ham_count = len(df[df['label'] == 'ham'])
p_spam = spam_count / total_messages
p_ham = ham_count / total_messages

print(f"\n1. Prior Probabilities:")
print(f"   P(spam) = {p_spam:.4f} ({spam_count}/{total_messages})")
print(f"   P(ham)  = {p_ham:.4f} ({ham_count}/{total_messages})")

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
train_df = pd.DataFrame({'message': X_train, 'label': y_train})

train_spam_messages = train_df[train_df['label'] == 'spam']
train_ham_messages = train_df[train_df['label'] == 'ham']
free_in_spam = train_spam_messages['message'].str.contains('free', case=False).sum()
free_in_ham = train_ham_messages['message'].str.contains('free', case=False).sum()

p_free_given_spam = free_in_spam / len(train_spam_messages)
p_free_given_ham = free_in_ham / len(train_ham_messages)

print(f"\n2. Likelihood Probabilities (from training data):")
print(f"   P('free'|spam) = {p_free_given_spam:.4f} ({free_in_spam}/{len(train_spam_messages)})")
print(f"   P('free'|ham)  = {p_free_given_ham:.4f} ({free_in_ham}/{len(train_ham_messages)})")

spam_score = p_free_given_spam * p_spam
ham_score = p_free_given_ham * p_ham

print(f"\n3. Posterior Analysis:")
print(f"   Spam Score (proportional to P(spam|'free')) = {spam_score:.4f}")
print(f"   Ham Score (proportional to P(ham|'free'))  = {ham_score:.4f}")
print("\nQuestion Answer: Based on these probabilities, a message with 'free' is more likely to be SPAM.\n")

print("--- Task 3: Building and Evaluating the Naive Bayes Classifier ---")

y_train_numeric = y_train.map({'ham': 0, 'spam': 1})
y_test_numeric = y_test.map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = nb.MultinomialNB()
model.fit(X_train_vectorized, y_train_numeric)

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test_numeric, y_pred)
conf_matrix = confusion_matrix(y_test_numeric, y_pred)

print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("Displaying Confusion Matrix...")

sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Spam Detection')
plt.show()