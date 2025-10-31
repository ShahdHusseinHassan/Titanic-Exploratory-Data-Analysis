import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

#View the First 5 Rows
print(df.head())

print(df.shape)
print(df.columns)
print(df.info())

#Check for Missing Values
print(df.isnull().sum())

#Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

print(df.isnull().sum())

#Summary Statistics
print(df.describe())

#Group-Based Insights
print(df.groupby('Sex')['Survived'].mean())
print(df.groupby('Pclass')['Survived'].mean())

#data viz
sns.countplot(x='Survived', hue='Survived',data=df, palette='Set1',legend=False)
plt.title('Survival Counts (0 = Died, 1 = Survived)')
plt.show()


sns.barplot(x='Sex', y='Survived', hue='Sex', data=df, palette='viridis', legend=False)
plt.title('Survival Rate by Gender')
plt.show()



# #Survival by Passenger Class viz
df_plot = df.copy()
df_plot['Pclass'] = df_plot['Pclass'].map({1: 'First Class', 2: 'Second Class',3: 'Third Class'})
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_plot, palette='Set2')
plt.title('Survival by Passenger Class', fontsize=14)
plt.show()
#
#
#
# #correlation heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
