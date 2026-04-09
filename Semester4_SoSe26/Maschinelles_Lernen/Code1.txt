
 %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from scipy.stats import chi2_contingency

# Download latest version
#https://www.kaggle.com/datasets/thedevastator/the-ultimate-netflix-tv-shows-and-movies-dataset
path = kagglehub.dataset_download("thedevastator/the-ultimate-netflix-tv-shows-and-movies-dataset")

print("Path to dataset files:", path)

# %% [markdown]
# Read data and first inspections

# %%
import pandas as pd

df = pd.read_csv("C:/Users/plassju/.cache/kagglehub/datasets/thedevastator/the-ultimate-netflix-tv-shows-and-movies-dataset/versions/2/Best Movies Netflix.csv")

# %%
df.head(5)

# %%
df.shape

# %%
df.columns

# %%
df.isna().sum()

# %% [markdown]
# First descriptive statistics

# %%
#alternativ df.info
df.dtypes

# %%
#Change objects to factors/categorys (recommendable for data analysis)
df['MAIN_GENRE'] = df['MAIN_GENRE'].astype('category')
df['MAIN_PRODUCTION'] = df['MAIN_PRODUCTION'].astype('category')
df['RELEASE_YEAR'] = df['RELEASE_YEAR'].astype('category')

# %%
#Values of a variable
set(df.MAIN_GENRE)

# %%
#wichtigste Kennzahlen fuer numerische Variablen
df.describe()

# %%
#Varianzberechnung
#Beide Methoden liefern ähnliche Ergebnisse, mit leichten Unterschieden aufgrund der Verwendung verschiedener Nenner:
#In numpy (ohne Korrektur) und in pandas ((n-1)-Korrektur)).
[np.var(df['SCORE']), df['SCORE'].var()]

# %%
#Kennzahlen fuer kategoriale Variablen
df.describe(include=['object', 'category'])
#dh Modus von MAIN_GENRE ist drama, von MAIN_PRODUCTION ist er US

# %%
#dichotomize variables
df['HIGH_SCORE'] = (df['SCORE']>7.5).astype(int)
df['MAIN_PRODUCTION_US'] = (df['MAIN_PRODUCTION'] == 'US').astype(int)

# %%
#Korrelationen (Pearson)
np.corrcoef(df.DURATION, df.SCORE)

# %%
data_crosstab = pd.crosstab(df['MAIN_PRODUCTION_US'],
                            df['HIGH_SCORE'], 
                               margins = False)
print(data_crosstab)

# %%

# Performing Chi-sq test
ChiSqResult = chi2_contingency(data_crosstab)
print('The Chi2-coefficient ist:', ChiSqResult[0], '\nThe P-Value of the ChiSq Test is:', ChiSqResult[1])
#dh H0 of independence can be rejected, there is a dependence on the associated significance level

# %%
Cramers_V = np.sqrt(ChiSqResult[0]/df.shape[0])
Cramers_V
#0.042941347970639775, very small association

# %%
data_crosstab_genre_score = pd.crosstab(df['MAIN_GENRE'],
                            df['SCORE'], 
                               margins = False)
ChiSqResult = chi2_contingency(data_crosstab_genre_score)
print('The Chi2-coefficient ist:', ChiSqResult[0], '\nThe P-Value of the ChiSq Test is:', ChiSqResult[1])
Cramers_V = np.sqrt(ChiSqResult[0]/df.shape[0])
Cramers_V
#high correlation between MAIN_GENRE and Score


# %% [markdown]
# Univariate visualizations

# %%
plt.figure(figsize=(15,5))
sns.countplot(x='MAIN_GENRE', data=df)
plt.title("Main genre of Netflix films")
plt.show()

# %%
plt.figure(figsize=(10,5))
df['MAIN_GENRE'].value_counts().head(5).plot(kind='bar')
plt.title("Top 5 Genres")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.show()

# %%
plt.figure(figsize=(10,5))
df['MAIN_PRODUCTION'].value_counts().head(5).plot(kind='bar')
plt.title("Top 5 production countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.show()

# %%
plt.boxplot(df.DURATION) 
plt.title('Duration of Movies') 
plt.xlabel('')
plt.ylabel('Duration in minutes')
plt.show()

# %% [markdown]
# Multivariate visualizations

# %%
plt.figure(figsize=(15,5))
sns.boxplot(data=df, x="MAIN_GENRE", y="SCORE")

# %%
sns.boxplot(data=df, x="MAIN_PRODUCTION_US", y="SCORE")

# %%
plt.figure(figsize=(15,5))
sns.boxplot(data=df, x="MAIN_GENRE", y="SCORE")

# %%
sns.scatterplot(x = 'DURATION', y = 'SCORE', data = df)
#no real association between duration and score, pearson corrcoef was also rather small with 0.13



 
