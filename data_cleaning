import os
import pandas as pd

y_train_raw = pd.read_csv("/home/f/jj2157/train_metadata").set_index('Accession').sort_index().dropna()
y_train_raw = y_train_raw.drop(['Country', 'Region'], axis = 1).dropna()
y_test_raw = pd.read_csv("/home/f/jj2157/test_metadata").set_index('Accession').sort_index().dropna()
y_test_raw = y_test_raw.drop(['Country', 'Region'], axis = 1).dropna()

train_pt = y_train_raw
test_pt = y_test_raw

train_pt['PT'] = train_pt['PT'].str.strip().str.removeprefix('PT').str.split().str.get(-1)
test_pt['PT'] = test_pt['PT'].str.strip().str.split().str.get(-1)
  
test_pt['PT'] = test_pt['PT'].replace(to_replace="21/28", value="2128")

train_pt = train_pt[train_pt['PT'] != '70']
train_pt = train_pt[train_pt['PT'] != '51']
train_pt = train_pt[train_pt['PT'] != 'none']
train_pt = train_pt[train_pt['PT'] != '24']
train_pt = train_pt[train_pt['PT'] != '74']
train_pt = train_pt[train_pt['PT'] != '87']
train_pt = train_pt[train_pt['PT'] != '42']
train_pt = train_pt[train_pt['PT'] != '63']
train_pt = train_pt[train_pt['PT'] != '90']
train_pt = train_pt[train_pt['PT'] != '88']
train_pt = train_pt[train_pt['PT'] != '10']
train_pt = train_pt[train_pt['PT'] != '38']
train_pt = train_pt[train_pt['PT'] != '91']
train_pt = train_pt[train_pt['PT'] != '1S']
train_pt = train_pt[train_pt['PT'] != '43']
train_pt = train_pt[train_pt['PT'] != '33']
train_pt = train_pt[train_pt['PT'] != 'untypable']
train_pt = train_pt[train_pt['PT'] != '46']
train_pt = train_pt[train_pt['PT'] != '56']
train_pt = train_pt[train_pt['PT'] != '23']
train_pt = train_pt[train_pt['PT'] != '67']
train_pt = train_pt[train_pt['PT'] != 'RDNC']

test_pt = test_pt[test_pt['PT'] != '45']
test_pt = test_pt[test_pt['PT'] != '89']  
test_pt = test_pt[test_pt['PT'] != '33']
test_pt = test_pt[test_pt['PT'] != 'untypable']
test_pt = test_pt[test_pt['PT'] != '46']
test_pt = test_pt[test_pt['PT'] != '56']
test_pt = test_pt[test_pt['PT'] != '23']
test_pt = test_pt[test_pt['PT'] != '67']
test_pt = test_pt[test_pt['PT'] != 'RDNC']

trainsampnames = train_pt.index.to_list()
testsampnames = test_pt.index.to_list()

X_train_raw = pd.read_csv("/lifesciences/pra-0104/jj2157/assembly_kmer_RF_STEC_14-18.txt", compression = 'gzip', sep = '\t').T
X_test_raw = pd.read_csv("/lifesciences/pra-0104/jj2157/assembly_kmer_RF_STEC_19.txt", compression = 'gzip', sep = '\t').T

new_samples = X_train_raw.index
new_samples = new_samples.to_list()
new_samples.pop(0)

X_train_raw = X_train_raw.iloc[:-1, :]
X_train_raw.index = new_samples
  
commonkmers = list(set(X_train_raw.columns.tolist()) & set(X_test_raw.columns.tolist()))
commontrainsampnames = list(set(X_train_raw.index.tolist()) & set(trainsampnames))
commontestsampnames = list(set(X_test_raw.index.tolist()) & set(testsampnames))

X_train = X_train_raw.loc[trainsampnames, commonkmers]
X_test = X_test_raw.loc[testsampnames, commonkmers]
train_pt = train_pt.loc[trainsampnames]
test_pt = test_pt.loc[testsampnames]

train_pt = pd.DataFrame(train_pt).sort_index()
test_pt = pd.DataFrame(test_pt).sort_index()

X_train = pd.DataFrame(X_train).sort_index()
X_test = pd.DataFrame(X_test).sort_index()

output_folder = "/lifesciences/pra-0104/jj2157"
os.makedirs(output_folder, exist_ok=True)

train_pt.to_csv(os.path.join(output_folder, "finalPTREALRAW_train_pt.csv"))
test_pt.to_csv(os.path.join(output_folder, "finalPTREALRAW_test_pt.csv"))
X_train.to_pickle(os.path.join(output_folder, "finalPTREALRAW_X_train.pkl"))
X_test.to_pickle(os.path.join(output_folder, "finalPTREALRAW_X_test.pkl"))



















import os
import pandas as pd

y_train_raw = pd.read_csv("/home/f/jj2157/train_metadata").set_index('Accession').sort_index().dropna()
y_train_raw = y_train_raw.drop(['Country', 'Region'], axis = 1).dropna()
y_test_raw = pd.read_csv("/home/f/jj2157/test_metadata").set_index('Accession').sort_index().dropna()
y_test_raw = y_test_raw.drop(['Country', 'Region'], axis = 1).dropna()

train_stx = y_train_raw
test_stx = y_test_raw

train_stx['Stx'] = train_stx['Stx'].str.strip()
test_stx['Stx'] = test_stx['Stx'].str.strip()

train_stx['Stx'] = train_stx['Stx'].replace(to_replace = "stx2c stx1a", value = "stx1a stx2c")
train_stx['Stx'] = train_stx['Stx'].replace(to_replace = "stx2a stx2c stx1a", value = "stx1a stx2a stx2c")
train_stx['Stx'] = train_stx['Stx'].replace(to_replace = "stx2a stx1a", value = "stx1a stx2a")

test_stx['Stx'] = test_stx['Stx'].replace(to_replace = "stx2c stx1a", value = "stx1a stx2c")
test_stx['Stx'] = test_stx['Stx'].replace(to_replace = "stx2a stx1a", value = "stx1a stx2a")
test_stx['Stx'] = test_stx['Stx'].replace(to_replace = "stx2a stx2c stx1a", value = "stx1a stx2a stx2c")

train_stx = train_stx[train_stx['Stx'] != 'none']
test_stx = test_stx[test_stx['Stx'] != 'stx1a stx1c stx2c']
test_stx = test_stx[test_stx['Stx'] != 'stx1a stx2c stx2d']



trainsampnames = train_stx.index.to_list()
testsampnames = test_stx.index.to_list()

X_train_raw = pd.read_csv("/lifesciences/pra-0104/jj2157/assembly_kmer_RF_STEC_14-18.txt", compression = 'gzip', sep = '\t').T
X_test_raw = pd.read_csv("/lifesciences/pra-0104/jj2157/assembly_kmer_RF_STEC_19.txt", compression = 'gzip', sep = '\t').T

new_samples = X_train_raw.index
new_samples = new_samples.to_list()
new_samples.pop(0)

X_train_raw = X_train_raw.iloc[:-1, :]
X_train_raw.index = new_samples

commonkmers = list(set(X_train_raw.columns.tolist()) & set(X_test_raw.columns.tolist()))
commontrainsampnames = list(set(X_train_raw.index.tolist()) & set(trainsampnames))
commontestsampnames = list(set(X_test_raw.index.tolist()) & set(testsampnames))

X_train = X_train_raw.loc[trainsampnames, commonkmers]
X_test = X_test_raw.loc[testsampnames, commonkmers]
train_stx = train_stx.loc[trainsampnames]
test_stx = test_stx.loc[testsampnames]

train_stx = pd.DataFrame(train_stx).sort_index()
test_stx = pd.DataFrame(test_stx).sort_index()

X_train = pd.DataFrame(X_train).sort_index()
X_test = pd.DataFrame(X_test).sort_index()

output_folder = "/lifesciences/pra-0104/jj2157"
os.makedirs(output_folder, exist_ok=True)

train_stx.to_csv(os.path.join(output_folder, "REALRAW_train_stx.csv"))
test_stx.to_csv(os.path.join(output_folder, "REALRAW_test_stx.csv"))
X_train.to_pickle(os.path.join(output_folder, "REALRAW_X_train.pkl"))
X_test.to_pickle(os.path.join(output_folder, "REALRAW_X_test.pkl"))
