import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataframe = pd.read_csv("/home/requiem/Documents/programs/AI&ML/homework/linregbostonhousing/housing.csv", header=None, delimiter=r"\s+", names=column_names)
print ("This dataset has these many features: ",len(dataframe.columns))

print("The dataset has a size of: ",np.shape(dataframe))

print("Stats of the dataset: \n",dataframe.describe())

#histogram plot of prices
plt.figure(figsize= (12,50))
histogram = sb.histplot(dataframe['MEDV'],)

#correlation matrix features and prices
numeric_df = dataframe.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()

plt.figure(figsize= (12,50))
mask = np.triu(np.ones_like(corr_matrix, dtype= bool))
cmap = sb.diverging_palette(210, 40, as_cmap = True)
sb.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix of Features', fontsize=16)

#scatter plot
plt.figure(figsize= (12,50))
sb.scatterplot(data=dataframe, x='MEDV', y='RM')
plt.title('Scatter Plot of Price to RM',fontsize = 12)


plt.show()

#linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = dataframe[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = dataframe['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 

print(f"R-squared (R2) score: {r2:.2f}")
if r2>0.4:
    print ("The model's performance is good.")
else:
    print("Your model's performance is bad.")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Predicted vs Actual')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.title('Actual vs. Predicted Housing Prices')
plt.legend()
plt.grid(True)
plt.show()