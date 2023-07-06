import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

# Ploting
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from tabulate import tabulate

#sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.impute import SimpleImputer

#imblearn
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Clasifiers
import xgboost
import inspect
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Scipy
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, skew, kurtosis

from collections import defaultdict
from tabpfn import TabPFNClassifier
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

trainDf = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
greeksDf = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
testDf = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
mergedDf = pd.merge(trainDf, greeksDf, how = 'left', on = 'Id')

trainDf.columns = trainDf.columns.str.strip()
testDf.columns = testDf.columns.str.strip()
catCols = 'EJ' 

numColsMerged = mergedDf.columns.tolist()[1:-1]
numColsMerged.remove(catCols)

greeksDf.tail(3)

Alpha = greeksDf['Alpha'].unique()
Beta = greeksDf['Beta'].unique()
Gamma = greeksDf['Gamma'].unique()
Delta = greeksDf['Delta'].unique()

import matplotlib.pyplot as plt

Alpha = greeksDf['Alpha'].value_counts()
Beta = greeksDf['Beta'].value_counts()
Gamma = greeksDf['Gamma'].value_counts()
Delta = greeksDf['Delta'].value_counts()

fig, axs = plt.subplots(1, 4, figsize=(12, 4))

axs[0].bar(Alpha.index, Alpha.values)
axs[0].set_title('Alpha')

axs[1].bar(Beta.index, Beta.values)
axs[1].set_title('Beta')

axs[2].bar(Gamma.index, Gamma.values)
axs[2].set_title('Gamma')

axs[3].bar(Delta.index, Delta.values)
axs[3].set_title('Delta')

plt.tight_layout()

plt.show()

len(greeksDf["Id"])

testDf.head(3)

len(trainDf["Id"])

counts = mergedDf['Class'].value_counts()
labels = counts.index.tolist()
sizes = counts.values.tolist()
colors = sns.color_palette('Set3')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'width': 0.35})
ax1.axis('equal')

y_pos = np.arange(len(labels))
ax2.barh(y_pos, sizes, color=colors)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Count')

plt.show()

# Min-max normalization
def minmax_scalar(df_train_in, df_valid_in, cols):
    """
    Applies min-max scaling to selected columns
    Args:
        df_train_in (DataFrame, shape (m, n)): input training dataframe
        df_valid_in (DataFrame, shape (m, n)): input validation dataframe
        cols (array_like, shape (r, ))       : list of columns to be normalized (r <= n)
        
    Returns:
        df_train_out (DataFrame, shape (m, n)): output training dataframe
        df_valid_out (DataFrame, shape (m, n)): output validation dataframe
    """
    df_train_out, df_valid_out = df_train_in.copy(deep = True), df_valid_in.copy(deep = True)
    cols = [col for col in cols if col in df_train_in.columns]
    cols = [col for col in cols if df_train_in[col].nunique() > 1]
    for col in cols:
        min_, max_ = df_train_out[col].min(), df_train_out[col].max()
        df_train_out[col] = (df_train_out[col] - min_) / (max_ - min_)
        df_valid_out[col] = (df_valid_out[col] - min_) / (max_ - min_)
    return df_train_out, df_valid_out

# Cross validation scores
def cv_scores(model, X, y, n_splits = 5, scaling = []):
    """
    Function to return cross validation log loss scores along with mean and standard deviation
    Args:
        model                            : untrained model
        X (DataFrame, shape (m, n))      : feature dataframe
        y (Series, shape (m, ))          : target variable
        n_splits (scalar)                : number of folds for cross validation
        scaling (array_like, shape (r, )): list of columns of X to be scaled (r <= n)
        
    Returns:
        scores (array_like, shape (n_splits, )): list of cross validation log loss scores
        mean (scalar): mean of the cross validation log loss scores
        std (scalar) : standard deviation of the cross validation log loss scores
    """
    scores = []
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 0)
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
        X_train, X_valid = minmax_scalar(X_train, X_valid, scaling)
        if 'CatBoostClassifier' in str(model):
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose = 0)
        else:
            model.fit(X_train, y_train)
        val_preds = model.predict_proba(X_valid)
        val_score = log_loss(y_valid, val_preds)
        scores.append(val_score)
    mean, std = np.mean(scores), np.std(scores)
    return scores, mean, std

# Categorical data encoding
le = LabelEncoder()
categorical_columns = mergedDf.columns[mergedDf.dtypes == 'object']
categorical_columns = [col for col in categorical_columns if col not in ['Id', 'Epsilon']]

mergedDf[categorical_columns] = mergedDf[categorical_columns].apply(lambda x: le.fit_transform(x))
mergedDf[categorical_columns].head(3)

# Columns to be scaled
scl = [col for col in mergedDf.columns if col not in ['Id', 'Class', 'Epsilon']]
scl = [col for col in scl if mergedDf[col].dtypes == 'float64']

featuresStd = mergedDf.loc[:,numColsMerged].apply(lambda x: np.std(x)).sort_values(
    ascending=False)
f_std = mergedDf[featuresStd.iloc[:20].index.tolist()]
f_std

featuresStd = mergedDf.loc[:,numColsMerged].apply(lambda x: np.std(x)).sort_values(
    ascending=False)
fStd = mergedDf[featuresStd.iloc[:20].index.tolist()]

with pd.option_context('mode.use_inf_as_na', True):
    featuresSkew = np.abs(mergedDf.loc[:,numColsMerged].apply(lambda x: np.abs(skew(x))).sort_values(
        ascending=False)).dropna()
skewed = mergedDf[featuresSkew.iloc[:20].index.tolist()]

with pd.option_context('mode.use_inf_as_na', True):
    featuresKurt = np.abs(mergedDf.loc[:,numColsMerged].apply(lambda x: np.abs(kurtosis(x))).sort_values(
        ascending=False)).dropna()
kurtF = mergedDf[featuresKurt.iloc[:20].index.tolist()]

def feat_dist(df, cols, rows=3, columns=3, title=None, figsize=(30, 25)):
    
    fig, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, j in zip(cols, axes):
        sns.kdeplot(df, x=i, ax=j, hue='Class', linewidth=1.5, linestyle='--')
        
        (mu, sigma) = norm.fit(df[i])
        
        xmin, xmax = j.get_xlim()[0], j.get_xlim()[1]
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        j.plot(x, p, 'k', linewidth=2)
        
        if 'Skewed' in title:
            label = 'Skewed'
        elif 'Kurtosis' in title:
            label = 'Kurtosis'
        elif 'Desviation Standard' in title:
            label = 'Desviation Standard'
        else:
            label = 'Normal Dist'
        
        j.set_title('Dist of {0} Norm Fit: $\mu=${1:.2g}, $\sigma=${2:.2f}'.format(i, mu, sigma), weight='bold')
        j.legend(labels=[f'Class0_{i}', f'Class1_{i}', label])
        fig.suptitle(f'{title}', fontsize=24, weight='bold')

feat_dist(mergedDf, f_std.columns.tolist(), rows=2, columns=4, title='High Desviation Standard Features', figsize=(30, 9))

feat_dist(mergedDf, skewed.columns.tolist(), rows=2, columns=4, title='Distribution of Skewed Features', figsize=(30, 9))

feat_dist(mergedDf, kurtF.columns.tolist(), rows=2, columns=4, title='Distribution of High Kurtosis Features', figsize=(30, 9))

correlations = mergedDf.loc[:,numColsMerged].corrwith(mergedDf['Class']).to_frame()
correlations['Abs Corr'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(sorted_correlations.iloc[1:].to_frame()[sorted_correlations>=.15], cmap='inferno', annot=True, vmin=-1, vmax=1, ax=ax)
plt.title('Feature Correlations With Target')
plt.show()

correlations = mergedDf.loc[:,numColsMerged].corr().abs().unstack().sort_values(kind="quicksort",ascending=False).reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']] #preventing 1.0 corr
corr_max=correlations.level_0.head(150).tolist()
corr_max=list(set(corr_max)) #removing duplicates

corr_min=correlations.level_0.tail(34).tolist()
corr_min=list(set(corr_min)) #removing duplicates
correlation_train = mergedDf.loc[:,corr_max].corr()

mask = np.triu(correlation_train.corr())

plt.figure(figsize=(30, 12))
sns.heatmap(correlation_train,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            linewidths=0.00,
            cbar=True)


plt.suptitle('Features with Highest Correlations',  weight='bold')
plt.tight_layout()

import os
filepath = '/kaggle/input/icr-identify-age-related-conditions'

correlations = mergedDf.loc[:,numColsMerged].corr().abs().unstack().reset_index()
correlations

fig, axes = plt.subplots(2, 4, figsize=(12,4), constrained_layout=True)
axes = axes.flatten()

# for i, j in zip(cols, axes):
i = 0
for row in range(0,16,2):
    a = correlations.reset_index(drop=True).loc[row, ['level_0', 'level_1']][0]
    b = correlations.reset_index(drop=True).loc[row, ['level_0', 'level_1']][1]    
   
    sns.regplot(mergedDf, x=a, y=b, ci=False, ax=axes[i], order=1, scatter_kws={'color':'red', 's':1.5}, line_kws={'color':'black', 'linewidth':1.5})
    i+=1
    
plt.suptitle('Highly Correlated Features',  weight='bold')
plt.show()

trainDf['EJ'] = trainDf['EJ'].replace({'A': 0, 'B': 1})

def hierarchical_clustering(data):
    fig, ax = plt.subplots(1, 1, figsize=(18, 10), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=data.columns, ax=ax, above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(['#000000'])  # Change color of lines to black
    plt.grid(axis='x')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=18, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Distance')
    plt.xticks(rotation=90)  # Rotate the column names for better readability.
    plt.show()

hierarchical_clustering(trainDf.drop(['Class', 'Id'], axis=1))

numericColumns = trainDf.select_dtypes(include=[np.number])
numericColumns = numericColumns.apply(pd.to_numeric, errors='coerce')
filteredData = numericColumns.loc[:, (numericColumns >= 0).all()]
filteredData = numericColumns[(numericColumns >= 0) & (numericColumns <= 1)]
filteredData = filteredData.dropna(axis=1, how='all')

pd.set_option('display.max_columns', None)
filteredData.head(2)

numericColumns = trainDf.select_dtypes(include=[np.number])
numericColumns = numericColumns.apply(pd.to_numeric, errors='coerce')

filteredData = numericColumns.loc[:, (numericColumns >= 50).all()]
filteredData = filteredData.loc[:, (filteredData <= 100).all()]
filteredData = numericColumns[(numericColumns >= 50) & (numericColumns <= 100)] 
filteredData =  filteredData.dropna(axis=1, how='all')

filteredData.head(2)

# Features target split
features = trainDf.drop(['Id', 'Class', 'BD', 'CD', 'CW', 'FD'], axis = 1).columns.tolist()
X, y = mergedDf[features], mergedDf['Class']

# XGBoost

xgb = XGBClassifier(n_jobs=-1)
xgb_scores_t, xgb_mean_t, xgb_std_t = cv_scores(xgb, X, y, n_splits=5, scaling=scl)
data = [['XGBoost', xgb_scores_t, xgb_mean_t, xgb_std_t]]
headers = ['Model', 'Scores', 'Mean', 'Std.Dev.']
dfXgb_scores_t = pd.DataFrame(data, columns=headers)
dfXgb_scores_t.head()

# CatBoost
catb = CatBoostClassifier()
catb_scores_t, catb_mean_t, catb_std_t = cv_scores(catb, X, y, n_splits=5, scaling=scl)
data = [['CatBoost', catb_scores_t, catb_mean_t, catb_std_t]]
headers = ['Model', 'Scores', 'Mean', 'Std.Dev.']
dfCatb = pd.DataFrame(data, columns=headers)
dfCatb.head()

# Features target split
features = mergedDf.drop(['Id', 'Class', 'Alpha', 'Epsilon'], axis = 1).columns.tolist()
X, y = mergedDf[features], mergedDf['Class']

# XGBoost
xgb = XGBClassifier(n_jobs=-1)
xgb_scores_tg, xgb_mean_tg, xgb_std_tg = cv_scores(xgb, X, y, n_splits=5, scaling=scl)
model = "XGBoost"
headers = ["Model", "Scores", "Mean", "Std.Dev."]
data = [[model, xgb_scores_tg, xgb_mean_tg, xgb_std_tg]]
dfXgb_scores_tg = pd.DataFrame(data, columns=headers)
dfXgb_scores_tg.head()

# CatBoost
catb = CatBoostClassifier()
catb_scores_tg, catb_mean_tg, catb_std_tg = cv_scores(catb, X, y, n_splits=5, scaling=scl)
model = "CatBoost"
headers = ["Model", "Scores", "Mean", "Std.Dev."]
data = [[model, catb_scores_tg, catb_mean_tg, catb_std_tg]]
dfCatb_scores_tg = pd.DataFrame(data, columns=headers)
dfCatb_scores_tg.head()

## Lets reset trainDf
trainDf = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')

FEATURE_COLUMNS = [i for i in trainDf.columns if i not in ["Id"]]

# Creates a GroupKFold with 5 splits
kf = KFold(n_splits=5)

# Create list of ids for the creation of oof dataframe.
ID_LIST = trainDf.index

# Create a dataframe of required size with zero values.
oof = pd.DataFrame(data=np.zeros((len(ID_LIST),1)), index=ID_LIST)

# Create an empty dictionary to store the models trained for each fold.
models = {}

# Create empty dict to save metircs for the models trained for each fold.
accuracy = {}
cross_entropy = {}

# Save the name of the label column to a variable.
label = "Class"

tfdf.keras.get_all_models()

# Calculate the number of negative and positive values in `Class` column
neg, pos = np.bincount(trainDf['Class'])
# Calculate total samples
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# Calculate the number of samples for each label.
neg, pos = np.bincount(trainDf['Class'])

# Calculate total samples.
total = neg + pos

# Calculate the weight for each label.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# Loop through each fold
for i, (train_index, valid_index) in enumerate(kf.split(X=trainDf)):
        print('##### Fold',i+1)

        # Fetch values corresponding to the index 
        train_df = trainDf.iloc[train_index]
        valid_df = trainDf.iloc[valid_index]
        valid_ids = valid_df.index.values
        
        # Select only feature columns for training.
        train_df = train_df[FEATURE_COLUMNS]
        valid_df = valid_df[FEATURE_COLUMNS]
        
        # There's one more step required before we can train the model. 
        # We need to convert the datatset from Pandas format (pd.DataFrame)
        # into TensorFlow Datasets format (tf.data.Dataset).
        # TensorFlow Datasets is a high performance data loading library 
        # which is helpful when training neural networks with accelerators like GPUs and TPUs.
        # Note: Some column names contains white spaces at the end of their name, 
        # which is non-comaptible with SavedModels save format. 
        # By default, `pd_dataframe_to_tf_dataset` function will convert 
        # this column names into a compatible format. 
        # So you can safely ignore the warnings related to this.
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
        valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df, label=label)

        # Define the model and metrics
        rf = tfdf.keras.RandomForestModel()
        rf.compile(metrics=["accuracy", "binary_crossentropy"]) 
        
        # Train the model
        # We will train the model using a one-liner.
        # Note: you may see a warning about Autograph. 
        # You can safely ignore this, it will be fixed in the next release.
        # Previously calculated class weights is used to handle imbalance.
        rf.fit(x=train_ds, class_weight=class_weight)
        
        # Store the model
        models[f"fold_{i+1}"] = rf
        
        
        # Predict OOF value for validation data
        predict = rf.predict(x=valid_ds)
        
        # Store the predictions in oof dataframe
        oof.loc[valid_ids, 0] = predict.flatten() 
        
        # Evaluate and store the metrics in respective dicts
        evaluation = rf.evaluate(x=valid_ds,return_dict=True)
        accuracy[f"fold_{i+1}"] = evaluation["accuracy"]
        cross_entropy[f"fold_{i+1}"]= evaluation["binary_crossentropy"]

tfdf.model_plotter.plot_model_in_colab(models['fold_1'], tree_idx=0, max_depth=3)

figure, axis = plt.subplots(3, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, fold_no in enumerate(models.keys()):
    row = i//2
    col = i % 2
    logs = models[fold_no].make_inspector().training_logs()
    axis[row, col].plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    axis[row, col].set_title(f"Fold {i+1}")
    axis[row, col].set_xlabel('Number of trees')
    axis[row, col].set_ylabel('Loss (out-of-bag)')

axis[2][1].set_visible(False)
plt.show()

for _model in models:
    inspector = models[_model].make_inspector()
    print(_model, inspector.evaluation())

average_loss = 0
average_acc = 0

for _model in  models:
    average_loss += cross_entropy[_model]
    average_acc += accuracy[_model]
    print(f"{_model}: acc: {accuracy[_model]:.4f} loss: {cross_entropy[_model]:.4f}")

print(f"\nAverage accuracy: {average_acc/5:.4f}  Average loss: {average_loss/5:.4f}")

inspector = models['fold_1'].make_inspector()

print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)

# Each line is: (feature name, (index of the feature), importance score)
inspector.variable_importances()["NUM_AS_ROOT"]

train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')

first_category = train.EJ.unique()[0]
train.EJ = train.EJ.eq(first_category).astype('int')
test.EJ = test.EJ.eq(first_category).astype('int')

def random_under_sampler(df):
    # Calculate the number of samples for each label. 
    neg, pos = np.bincount(df['Class'])

    # Choose the samples with class label `1`.
    one_df = df.loc[df['Class'] == 1] 
    # Choose the samples with class label `0`.
    zero_df = df.loc[df['Class'] == 0]
    # Select `pos` number of negative samples.
    # This makes sure that we have equal number of samples for each label.
    zero_df = zero_df.sample(n=pos)

    # Join both label dataframes.
    undersampled_df = pd.concat([zero_df, one_df])

    # Shuffle the data and return
    return undersampled_df.sample(frac = 1)


train_good = random_under_sampler(train)

train_good.shape

predictor_columns = [n for n in train.columns if n != 'Class' and n != 'Id']
x= train[predictor_columns]
y = train['Class']

from sklearn.model_selection import KFold as KF, GridSearchCV
cv_outer = KF(n_splits = 10, shuffle=True, random_state=42)
cv_inner = KF(n_splits = 5, shuffle=True, random_state=42)

def balanced_log_loss(y_true, y_pred):
    # y_true: correct labels 0, 1
    # y_pred: predicted probabilities of class=1
    # calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # calculate the weights for each class to balance classes
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    # calculate the predicted probabilities for each class
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    # calculate the summed log loss for each class
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    # calculate the weighted summed logarithmic loss
    # (factgor of 2 included to give same result as LL with balanced input)
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    # return the average log loss
    return balanced_log_loss/(N_0+N_1)

class Ensemble():
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        self.classifiers =[xgboost.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.2,subsample=0.9,colsample_bytree=0.85),
                           xgboost.XGBClassifier(),
                           TabPFNClassifier(N_ensemble_configurations=24),
                          TabPFNClassifier(N_ensemble_configurations=64)]
    
    def fit(self,X,y):
        y = y.values
        unique_classes, y = np.unique(y, return_inverse=True)
        self.classes_ = unique_classes
        first_category = X.EJ.unique()[0]
        X.EJ = X.EJ.eq(first_category).astype('int')
        X = self.imputer.fit_transform(X)
#         X = normalize(X,axis=0)
        for classifier in self.classifiers:
            if classifier==self.classifiers[2] or classifier==self.classifiers[3]:
                classifier.fit(X,y,overwrite_warning =True)
            else :
                classifier.fit(X, y)
     
    def predict_proba(self, x):
        x = self.imputer.transform(x)
#         x = normalize(x,axis=0)
        probabilities = np.stack([classifier.predict_proba(x) for classifier in self.classifiers])
        averaged_probabilities = np.mean(probabilities, axis=0)
        class_0_est_instances = averaged_probabilities[:, 0].sum()
        others_est_instances = averaged_probabilities[:, 1:].sum()
        # Weighted probabilities based on class imbalance
        new_probabilities = averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(averaged_probabilities.shape[1])]])
        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1) 

from tqdm.notebook import tqdm

def training(model, x,y,y_meta):
    outer_results = list()
    best_loss = np.inf
    split = 0
    splits = 5
    for train_idx,val_idx in tqdm(cv_inner.split(x), total = splits):
        split+=1
        x_train, x_val = x.iloc[train_idx],x.iloc[val_idx]
        y_train, y_val = y_meta.iloc[train_idx], y.iloc[val_idx]
                
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_val)
        probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
        p0 = probabilities[:,:1]
        p0[p0 > 0.86] = 1
        p0[p0 < 0.14] = 0
        y_p = np.empty((y_pred.shape[0],))
        for i in range(y_pred.shape[0]):
            if p0[i]>=0.5:
                y_p[i]= False
            else :
                y_p[i]=True
        y_p = y_p.astype(int)
        loss = balanced_log_loss(y_val,y_p)

        if loss<best_loss:
            best_model = model
            best_loss = loss
            print('best_model_saved')
        outer_results.append(loss)
        print('>val_loss=%.5f, split = %.1f' % (loss,split))
    print('LOSS: %.5f' % (np.mean(outer_results)))
    return best_model

from datetime import datetime
times = greeks.Epsilon.copy()
times[greeks.Epsilon != 'Unknown'] = greeks.Epsilon[greeks.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
times[greeks.Epsilon == 'Unknown'] = np.nan

train_pred_and_time = pd.concat((train, times), axis=1)
test_predictors = test[predictor_columns]
first_category = test_predictors.EJ.unique()[0]
test_predictors.EJ = test_predictors.EJ.eq(first_category).astype('int')
test_pred_and_time = np.concatenate((test_predictors, np.zeros((len(test_predictors), 1)) + train_pred_and_time.Epsilon.max() + 1), axis=1)


ros = RandomOverSampler(random_state=42)

train_ros, y_ros = ros.fit_resample(train_pred_and_time, greeks.Alpha)
print('Original dataset shape')
print(greeks.Alpha.value_counts())
print('Resample dataset shape')
print( y_ros.value_counts())

x_ros = train_ros.drop(['Class', 'Id'],axis=1)
y_ = train_ros.Class

yt = Ensemble()

m = training(yt,x_ros,y_,y_ros)

y_.value_counts()/y_.shape[0]

y_pred = m.predict_proba(test_pred_and_time)
probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
p0 = probabilities[:,:1]
p0[p0 > 0.74] = 1
p0[p0 < 0.26] = 0

submission = pd.DataFrame(test["Id"], columns=["Id"])
submission["class_0"] = p0
submission["class_1"] = 1 - p0
submission.to_csv('submission.csv', index=False)

submission_df = pd.read_csv('submission.csv')
submission_df

# Retur
    