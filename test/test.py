import pandas as pd
import sys
sys.path.insert(1, '../')
from rfImputer import rfImputer
import numpy as np
from pprint import pprint
cols = ['Proof_cut', 'Viscosity', 'Caliper', 'Ink_temperature', 'Humifity',
        'Roughness', 'Blade_pressure', 'Varnish_pct', 'Press_speed', 'Ink_pct',
        'Solvent_pct', 'Esa_voltage', 'ESA_amperage', 'Wax', 'Hardener',
        'Roller_durometer', 'Density', 'Anode_ratio', 'Chrome_content', 'bands']
df = pd.read_csv('bands.csv')
df.columns = cols

df.replace('?', 'NaN', inplace = True)
df.replace('band', '1', inplace = True)
df.replace('noband', '0', inplace = True)

for col in df:
    if col == 'bands':
        continue
    df[col] = df[col].astype(float)

# Generate fake data
# n = 100
# df = pd.DataFrame(index = np.arange(0, n))
# df['cont_3'] = np.random.randn(n)
# df['cont_4'] = np.random.randn(n)
# df['cat_1'] = pd.Categorical(np.random.randint(0, 2, n)) 
# df['cat_2'] = pd.Categorical(np.random.randint(0, 2, n))
# df['cat_3'] = pd.Categorical(np.random.randint(0, 2, n))
# df['cat_4'] = pd.Categorical(np.random.randint(0, 2, n))
# df['cont_1'] = np.random.randn(n) + 0.5 * df['cat_2'] + df['cat_3']
# df['cont_2'] = np.random.randn(n) + 0.5 * df['cont_1'] + df['cat_1']

#Fill in some more missing data
# n = df.shape[0]
# for col in df.columns:
#     n_missing = np.random.randint(low = 0, high = n/4, size = 1)[0]
#     missing_idx = np.random.choice(n, n_missing, replace = False)
#     df[col].iloc[missing_idx] = np.nan

imp_df = rfImputer(df)

imp_df.impute('random_forest', {'n_estimators': 100, 'n_jobs': 1})
print "IMPUTED"

out = imp_df.imputed_df()

print out.head()

