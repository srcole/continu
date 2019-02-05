import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Load predictions
df_pred_orig = pd.read_csv('../processed_data/lgbm_v4_predictions.csv', index_col=0)

# Reset ypred to validation-based calibration
df_pred_orig['adj_ypred'] = df_pred_orig['adj2_ypred']

# Only select data in which test_label is 1
df_pred = df_pred_orig[df_pred_orig['test_label'] == 1]
df_pred = df_pred[['user', 'sess', 'task', 'trial', 'adj_ypred']]
df_pred['adj_ypred'] = df_pred['adj_ypred'] * 100

# Define users to keep
users_see = np.concatenate([np.arange(1,5), np.arange(65,76)])

# Only use some users
df_temp = df_pred.groupby('user')['adj_ypred'].mean().reset_index()
df_temp = df_temp.sort_values(by='adj_ypred', ascending=False).reset_index(drop=True)
df_temp = df_temp.loc[users_see]
users_use = df_temp['user'].unique()
df_pred = df_pred[df_pred['user'].isin(users_use)]
df_pred['sess'] = df_pred['sess'].astype(int)
df_pred['task'] = df_pred['task'].astype(int)
df_pred['trial'] = df_pred['trial'].astype(int)

# Load features
df_feat_orig = pd.read_csv('../processed_data/lgbm_v4_feat_matrix.csv', index_col=0)
# Only keep appropriate users
df_feat = df_feat_orig[df_feat_orig['user'].isin(users_use)]
df_feat = df_feat[df_feat['sess'] == 2]

# Compute typical profile for each user
feat90 = df_feat_orig.groupby('user').quantile(.9).drop(['sess', 'task', 'trial'], axis=1)
feat10 = df_feat_orig.groupby('user').quantile(.1).drop(['sess', 'task', 'trial'], axis=1)
feat90 = feat90.rename_axis('feature',axis=1).stack().reset_index().rename(columns={0: 'pc90'})
feat10 = feat10.rename_axis('feature',axis=1).stack().reset_index().rename(columns={0: 'pc10'})
df_feat_stats = feat90.merge(feat10, on=['user', 'feature'])

# Only keep some users
df_feat_stats = df_feat_stats[df_feat_stats['user'].isin(users_use)]

# Load messages for each trial
df_msgs = pd.read_csv('../processed_data/messages_per_trial.csv', index_col=0)
df_msgs = df_msgs[df_msgs['user'].isin(users_use)]
df_msgs = df_msgs[df_msgs['sess'] == 2]

# Add a hack to one user
# Define the hack
user_breached = 8
user_enemy = 16
N_trials_breached = 4
fake_msgs = ['watch this unbelievable video at fakewebsite.com',
             'fifteen million dollars in a foreign savings account',
             'get rich quick by entering your credit card here',
             'millenials can get free debt relief from this company']
assert len(fake_msgs) == N_trials_breached

# Add hacked keystrokes
df_temp = df_pred_orig[df_pred_orig['user']==user_breached]
df_temp = df_temp[df_temp['real_user'] == user_enemy]
df_temp = df_temp.head(N_trials_breached)
df_temp['sess'] = 3
df_temp.drop(['ypred', 'test_label', 'real_user'], axis=1, inplace=True)
df_pred = pd.concat([df_pred, df_temp])

df_temp = df_feat_orig[df_feat_orig['user']==user_enemy]
df_temp = df_temp[df_temp['sess']==2]
df_temp = df_temp.head(N_trials_breached)
df_temp['sess'] = 3
df_temp['user'] = user_breached
df_feat = pd.concat([df_feat, df_temp])

df_temp = df_temp[['user', 'sess', 'task', 'trial']]
df_temp['msg'] = fake_msgs
df_msgs = pd.concat([df_msgs, df_temp])

# Merge trial data
df_trials = df_msgs.merge(df_pred, on=['user', 'sess', 'task', 'trial'])
df_trials = df_trials.merge(df_feat, on=['user', 'sess', 'task', 'trial'])

# Save to SQL
engine = create_engine('sqlite:///continu_dash.db', echo=False)
df_feat_stats.to_sql('users', con=engine)
df_trials.to_sql('messages', con=engine)
