import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('/home/requiem/Documents/programs/AI&ML/homework/Module3/EDA&feature_engineering not done yet/data.csv')


#Summarize missing values and data types
print("Data Info:")
df.info()

print("\nMissing Values Count:")
print(df.isnull().sum())

#Visualize distributions of key features
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distributions of Key Features', fontsize=16)

# Age distribution
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Age Distribution')

# Fare distribution
sns.histplot(df['Fare'], kde=True, ax=axes[0, 1], bins=30)
axes[0, 1].set_title('Fare Distribution')

# Pclass and Sex distributions
sns.countplot(x='Pclass', data=df, ax=axes[1, 0]).set_title('Passenger Class Count')
sns.countplot(x='Sex', data=df, ax=axes[1, 1]).set_title('Sex Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#Analyze relationships between features and survival
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Survival Rates by Feature', fontsize=16)

# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0]).set_title('Survival by Sex')

# Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[1]).set_title('Survival by Pclass')

plt.show()

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                'Jonkheer', 'Dona'], 'Rare')
# Standardize common titles
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

print("\nValue counts of new 'Title' feature:")
print(df['Title'].value_counts())

#Handle missing values
# Impute 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Impute 'Embarked' with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Impute 'Fare' with the median if there are any missing values
df['Fare'].fillna(df['Fare'].median(), inplace=True)

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

print("\nData head after cleaning and encoding:")
print(df.head())

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Check data readiness
print("\n--- Data is Ready for Modeling ---")
print(f"Feature matrix (X) shape: {X.shape}")
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")