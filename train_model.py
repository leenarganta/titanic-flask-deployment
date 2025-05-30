import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'embarked', 'fare', 'survived'])

df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])

X = df[['age', 'fare', 'sex', 'embarked']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'titanic_model.pkl')
print(" The Model is now saved as titanic_model.pkl")
