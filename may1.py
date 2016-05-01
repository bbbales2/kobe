#%%

import numpy
import sklearn.linear_model
import pandas
import os
import datetime
import re
import sklearn.cross_validation
import patsy
import statsmodels.api as sm

os.chdir('/home/bbales2/backup/kaggle_data/kobe')

df = pandas.DataFrame.from_csv('data.csv')

df['shot_index'] = range(len(df))

df.set_index('shot_index')
#%%
# Compute the time the shot was taken
elapsed_minutes = (df['period'] - 1) * 12 + (12 - (df['seconds_remaining'] / 60.0 + df['minutes_remaining']))
df['minutes2'] = elapsed_minutes.apply(numpy.floor).astype('int')
df['seconds2'] = ((elapsed_minutes - df['minutes2']) * 60).astype('int')
df['hours2'] = (df['minutes2'] / 60).astype('int')
df['minutes2'] = df['minutes2'] - df['hours2'] * 60
var = df.apply(lambda x : datetime.time(x['hours2'], x['minutes2'], x['seconds2']), axis = 1)
df['shot_time'] = var
df['game_date'] = df['game_date'].astype('datetime64[ns]')

shotTime = df.apply(lambda x : datetime.datetime.combine(x['game_date'], x['shot_time']), axis = 1)


df['shot_time'] = shotTime

#%%

filled = df[df['shot_made_flag'].apply(numpy.isnan) == False]
missing = df[df['shot_made_flag'].apply(numpy.isnan) == True]

#%%

def process(data):
    home = data['matchup'].apply(lambda x : not bool(re.search('@', x)))
    opponent = data['opponent']
    season = data['season']
    shotType = data['combined_shot_type']
    time = data['shot_time']
    index = data['shot_index']

    return pandas.DataFrame({ 'index' : index, 'home' : home, 'opponent' : opponent, 'season' : season, 'type' : shotType, 'time' : time })

processed = process(filled)
processed['made'] = filled['shot_made_flag'].astype('int')
processed_missing = process(missing)
#%%
train, test = sklearn.cross_validation.train_test_split(processed)

train_ys, train_xs = patsy.dmatrices("made ~ home + season + type", train)
test_ys, test_xs = patsy.dmatrices("made ~ home + season + type", test)

#%%


model = sm.Logit(ys, xs)

result = model.fit()

print result.summary()

#%%
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

out = result.predict(test_xs)

print logloss(test_ys[:, 0], out)

#%%
import time
results = []
processed_missing2 = processed_missing.sort(['time'])
for i, (index, row) in enumerate(processed_missing2.iterrows()):
    tmp = time.time()
    training = processed[processed['time'] < row['time']]

    if len(training) > 0:
        ys, xs = patsy.dmatrices("made ~ home + season + type", train)

        model = sm.Logit(ys, xs)

        result = model.fit()

        (missing_xs,) = numpy.asarray(patsy.build_design_matrices([xs.design_info], row))

        answer = result.predict(missing_xs)

        results.append((row['index'], answer[0]))
    else:
        results.append((row['index'], 0.4))
        print 'Missing training data', row['index']
    print i, '/', len(processed_missing), time.time() - tmp
#%%

#%%

full_ys, full_xs = patsy.dmatrices("made ~ home + season + type", processed)
missing_xs = patsy.dmatrix("home + season + type", processed_missing)

result = model.fit()

print result.summary()

out = result.predict(missing_xs)

