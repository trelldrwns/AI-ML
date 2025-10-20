import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


sns.set_style("whitegrid")

df = pd.read_csv('/home/requiem/Documents/programs/AI&ML/homework/Module3/logisticregression/Iris.csv', header = 0)

#inefficient way 
# species_data = []
# species = (df['Species'])
# for i in species:
#     if i == "Iris-setosa":
#         species_data.append(i)
# print(species_data)

insert_loc = df.columns.get_loc('Species') + 1

new_col_data = (df['Species'] == 'Iris-setosa').astype(int)

df.insert(loc=insert_loc, column='is_setosa', value=new_col_data)

print ("1 is setosa type and 0 is all others: \n", df['is_setosa'].value_counts())
print("Not Balanced.")

#plots 
#barplot
plt.figure(figsize= (12,50))
sns.countplot(x='is_setosa', data=df)
plt.title('Class Distribution (Setosa vs. Non-Setosa)')
plt.xlabel('Class (0 = Non-Setosa, 1 = Setosa)')
plt.ylabel('Frequency')

#scatterplot
plt.figure(figsize= (12,50))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='is_setosa', data=df)
plt.title('Petal Length vs. Petal Width')


#features mean
if 'Id' in df.columns:
    df_features = df.drop('Id', axis=1)
else:
    df_features = df.copy()

feature_means = df_features.groupby('is_setosa').mean(numeric_only=True)
print("Mean of each feature grouped by class:")
print(feature_means)


#training model
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]

y = df['is_setosa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel training complete.")

y_pred = model.predict(X_test)


X_2d = df[['PetalLengthCm', 'PetalWidthCm']]
y_2d = df['is_setosa']

model_2d = LogisticRegression()
model_2d.fit(X_2d, y_2d)

x_min, x_max = X_2d.iloc[:, 0].min() - 0.5, X_2d.iloc[:, 0].max() + 0.5
y_min, y_max = X_2d.iloc[:, 1].min() - 0.5, X_2d.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='is_setosa',
                data=df, edgecolor='k')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

#accuracy of model
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on the test set: {accuracy * 100:.2f}%")

#confusion matrix 
plt.figure(figsize= (12,50))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Setosa', 'Setosa'], 
            yticklabels=['Non-Setosa', 'Setosa'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.show()


#model accuracy is 100%
#confusion matrix shows true negative (the non setosa flowers which were identified), and then shows the bottom right which is the true positive(setosa flowers which were identified) the model predicts the non setosa flowers better. 
#most important cofficients are the petal width and petal length both of them have large negative coefficients which means that the larger the petal width and length gets the higher probabilty of it not being a setosa flower increases. 
#absolute coefficients here are the petal width and petal length.