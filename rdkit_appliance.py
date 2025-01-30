import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem

pd.set_option('display.max_columns', None)

PATH = "C:\\Users\\georg\\OneDrive\\Документы\\лаба\\papyrus\\data_tables\\P.tsv"
TARGET = '\\'.join((PATH.split('\\')[:-1])) + '\\' + 'Ps_ECFPs.txt'

df = pd.read_csv(PATH, sep='\t')
print('таблица прочитана')

# print(df.head())

result_dict = {}

table_dict = {k: v + 1 for v, k in enumerate(list(df))}
for c, line in enumerate(df.itertuples()):
    connectivity = line[table_dict['connectivity']]
    smiles = line[table_dict['SMILES']]
    if connectivity not in result_dict:
        result_dict[connectivity] = list(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),
                                                  radius=6,
                                                  nBits=1024)
        )
    count = c + 1
    if count % 10000 == 0:
        print(f'пролистано {count} строчек')

print(len(result_dict))
with open(TARGET, 'w') as file:
    json.dump(result_dict, file)
