import os, sys, time, copy
from pathlib import Path
from tqdm.auto import tqdm
tqdm.pandas()

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set matplotlib configuration
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define data directory
# DATA_DIR = Path("../input/icr-identify-age-related-conditions/")
DATA_DIR = Path("/Users/maxmender/Desktop/Kaggle/Disease/icr-identify-age-related-conditions")


# Load data
train_df = pd.read_csv(DATA_DIR/'train.csv')
greek_df = pd.read_csv(DATA_DIR/'greeks.csv')
test_df = pd.read_csv(DATA_DIR/'test.csv')
sample_df = pd.read_csv(DATA_DIR/'sample_submission.csv')
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Greek shape: {greek_df.shape}")

# Print summary of data
def summary(df):
    # Print shape
    print(f"Data shape: {df.shape}")
    
    # Print columns with missing values
    print("\nSummary of columns missing values")
    print(df[df.isna().sum()[df.isna().sum() > 0].index].dtypes)
    print()
    
    # Print summary statistics
    summ = pd.DataFrame(df.dtypes, columns=['Data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first quartile'] = desc.loc[:, '25%'].values
    summ['second quartile'] = desc.loc[:, '50%'].values
    summ['third quartile'] = desc.loc[:, '75%'].values 
    
    return summ

summary(train_df)

# Select numerical and categorical features
num_cols = test_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = test_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols.remove('Id')
print(f"Total numerical features: {len(num_cols)}")
print(f"Total categorical features: {len(cat_cols)}")

# Plot distribution of labels
df = pd.DataFrame(train_df['Class'].value_counts())
df['pct (%)'] = np.round((train_df['Class'].value_counts()/train_df.shape[0]).values * 100, 4)
print(df, "\n")
plt.title('Distribution of labels', fontsize=20)
train_df['Class'].value_counts().plot(kind='bar');

# Visualize features vs target
figsize = (4*4, 20)
fig = plt.figure(figsize=figsize)
for idx, col in enumerate(num_cols):
    ax = plt.subplot(11, 5, idx + 1)
    sns.kdeplot(
        data=train_df, hue='Class', fill=True,
        x=col, palette=['green', 'red'], legend=False
    )
    
    ax.set_ylabel(''); ax.spines['top'].set_visible(False)
    ax.set_xlabel(''); ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', weight='bold', fontsize=20)
    
fig.suptitle(f'Features Vs target\n\n\n', ha='center', fontweight='bold', fontsize=20)
fig.legend([0, 1], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=20, ncol=3)
plt.tight_layout()
plt.show()

# Visualize outliers
def visualize_outliers(df1, num_cols=num_cols):
    cols = 4
    rows = len(num_cols) // cols + 1
    plt.figure(figsize=(10,20))
    for idx, feature in enumerate(num_cols):
        ax = plt.subplot(rows, cols, idx+1)
        sns.boxplot(
            data=df1, x='Class', y=feature
        )
        ax.set_title(feature)
        ax.set_ylabel("Distribution")
        
    plt.tight_layout()
    plt.show()

visualize_outliers(train_df)

# Show histogram
def show_histogram(df1, num_cols=num_cols):
    cols = 4
    rows = len(num_cols) // cols + 1
    plt.figure(figsize=(10,20))
    for idx, feature in enumerate(num_cols):
        ax = plt.subplot(rows, cols, idx+1)
        sns.histplot(
            data=df1[feature], kde=True
        )
        ax.set_ylabel("Count")
        
    plt.tight_layout()
    plt.show()

show_histogram(train_df)

weird_features = ['AH', 'AR', 'AY',  'BC', 'BR', 'BR', 'CB', 'CL', 'DF', 'DU', 'DV', 'EH', 'EP', 'EU',
                 'FC', 'FD ', 'FL', 'FR', 'FS', 'GE']

train_df[weird_features].plot(subplots=True, figsize=(15, 15));

good_features = ['AZ', 'BN', 'BQ', 'CC', 'CD ', 'CF', 'CR', 'CU', 'CW ', 'DA', 'DE', 'DH', 'DL', 'DN', 'DY', 
                'EE', 'EL', 'FI', 'GB', 'GH', 'GI']

train_df[good_features].plot(subplots=True, figsize=(15, 15));

# Data scaling
std_scaler = preprocessing.StandardScaler()
mm_scaler = preprocessing.MinMaxScaler()
robust_scaler = preprocessing.RobustScaler()
quantile_scaler = preprocessing.QuantileTransformer(n_quantiles=60, output_distribution='uniform')
scaled_train_df = train_df.copy(deep=True)
scaled_test_df = test_df.copy(deep=True)
scaled_train_df[num_cols] = quantile_scaler.fit_transform(scaled_train_df[num_cols])
scaled_test_df[num_cols] = quantile_scaler.fit_transform(scaled_test_df[num_cols])

# Check outliers in the scaled data
visualize_outliers(scaled_train_df)
# Check the histogram
show_histogram(scaled_train_df)

more_weird_features = ['AR', 'BZ', 'DF', 'DV']
scaled_train_df[more_weird_features].plot(subplots=True, figsize=(15,15));

scaled_train_df.head()

# Split data into features (X) and target (y)
x = scaled_train_df.drop('Id', axis=1)
X = x.drop('Class', axis=1)
y = scaled_train_df[['Class']]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_column = 'EJ'  
label_encoder = LabelEncoder()
X_train[categorical_column] = label_encoder.fit_transform(X_train[categorical_column])
X_test[categorical_column] = label_encoder.transform(X_test[categorical_column])

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create models
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.005, random_state=42, loss='log_loss')
svm_model = SVC(kernel='rbf', C=1.0, probability=True, gamma='scale', random_state=42)
cat_params = {
        'learning_rate': 0.005, 
        'iterations': 50500, 
        'depth': 4,
        'colsample_bylevel': 0.50,
        'subsample': 0.80,
        'l2_leaf_reg': 3,
        'random_seed': 4,
        'auto_class_weights': 'Balanced',
        'verbose' : 1000
    }
gbc = CatBoostClassifier(**cat_params)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit models
gb_model.fit(X_train, np.ravel(y_train))
svm_model.fit(X_train, np.ravel(y_train))
gbc.fit(X_train, np.ravel(y_train))
rf.fit(X_train, np.ravel(y_train))

# Make predictions
gb_predictions = gb_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
cb_predictions = gbc.predict(X_test)
rf_predictions = rf.predict(X_test)

# Soft Voting (combines predicted probabilities)
ensemble_model = VotingClassifier(estimators=[('gb', gb_model), ('svm', svm_model), ('cat', gbc), ('rf', rf)], voting='soft')

# Fit ensemble model
ensemble_model.fit(X_train, np.ravel(y_train))

# Make predictions with ensemble model
ensemble_predictions = ensemble_model.predict(X_test)

# Calculate accuracies
gb_accuracy = accuracy_score(y_test, gb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
cat_accuracy = accuracy_score(y_test, cb_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

# Print accuracies
print(f"Gradient Boosting Classifier Accuracy: {gb_accuracy}")
print(f"SVM Classifier Accuracy: {svm_accuracy}")
print(f"CatBoost Classifier Accuracy: {cat_accuracy}")
print(f"Random Forest Classifier Accuracy: {rf_accuracy}")
print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

