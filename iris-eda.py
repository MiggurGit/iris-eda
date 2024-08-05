import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset('iris')


print("First few rows of the Iris dataset:")
print(iris.head())


print("\nSummary statistics of the Iris dataset:")
print(iris.describe())


print("\nMissing values in the Iris dataset:")
print(iris.isnull().sum())


sns.pairplot(iris)
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()


plt.figure(figsize=(12, 6))
for i, feature in enumerate(iris.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris)
    plt.title(f'Boxplot of {feature} by Species')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, feature in enumerate(iris.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(data=iris, x=feature, hue='species', fill=True)
    plt.title(f'KDE plot of {feature} by Species')
plt.tight_layout()
plt.show()


sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='species')
plt.title('Scatter plot of Petal Length vs Petal Width')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

