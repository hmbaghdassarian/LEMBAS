#!/usr/bin/env python
# coding: utf-8

# Here, we preprocess the [immune dictionary](https://doi.org/10.1038/s41586-023-06816-9) dataset as described in scPerturb in anticipation that datasets from scPerturb will also be used. rds files are downloaded from the Immune Dictionary [download](https://www.immune-dictionary.org/app/home) page.

# In[1]:


import os

import scanpy as sc
import anndata as ad

import warnings
warnings.filterwarnings(action='ignore', module='pandas')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='scanpy')
warnings.filterwarnings(action='ignore', category=UserWarning, module='scanpy')

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))
from preprocess import get_tf_activity


# In[2]:


n_cores = 8
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'


# In[3]:


quick_run = False # whether to subsample data and use quicker/less mem intensive (less accurate) versions of parameters

iter_dict = {'quick': {'perm': int(10), 'n_samples': int(1e3), 'batch_size': int(1e3)},
             'full': {'perm': int(1e3), 'n_samples': None, 'batch_size': int(1e4)}}
if quick_run: 
    run_key = 'quick'
else:
    run_key = 'full'


# In[4]:


# grid search params
use_raw = False
# impute = True


# In[5]:


directory_names = [
    #data_path,
    #os.path.join(data_path, 'raw'), os.path.join(data_path, 'raw', 'immune_dictionary'),
    os.path.join(data_path, 'interim'), os.path.join(data_path, 'interim', 'immune_dictionary_h5ad'),
    os.path.join(data_path, 'processed'), os.path.join(data_path, 'figures')
]

for directory_name in directory_names:
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


# Load h5ad files:

# In[6]:


h5ad_in_path = os.path.join(data_path, 'interim', 'immune_dictionary_h5ad')
file_names = os.listdir(h5ad_in_path)

imm_d = []
for file_name in file_names:
    cell_type = file_name.split('ref_data_')[1].split('.h5ad')[0]
    adata = sc.read_h5ad(os.path.join(h5ad_in_path, file_name)) # Seurat counts slot in adata.raw.X, data slot in adata.X
    # imm_d[cell_type] = adata
    imm_d.append(adata)
adata = ad.concat(imm_d,  join="outer")


# In[7]:


adata


# In[8]:


sc.pl.tsne(adata, color='celltype')


# In[9]:


adata


# Get TF activity estimates:

# In[10]:


if quick_run:
    adata = sc.pp.subsample(adata, n_obs = iter_dict[run_key]['n_samples'], copy = True) # for quick tests

kwargs = {'args' : {'wsum' : {'times': iter_dict[run_key]['perm'], 'batch_size': iter_dict[run_key]['batch_size']},
                       'ulm' : {'batch_size': iter_dict[run_key]['batch_size']}, 
                        'mlm': {'batch_size': iter_dict[run_key]['batch_size']}
                       }}
# kwargs['methods'] = ['lm', 'ulm', 'wsum', 
# kwargs['cns_metds'] = ['lm', 'ulm', 'wsum_norm']
get_tf_activity(adata, organism = 'mouse', grn = 'collectri', verbose = True, 
                min_n = 5, use_raw = use_raw, **kwargs)


# save
fn = 'ID_TF_activity_estimate_' + run_key + '.csv'
fn = os.path.join(data_path, 'interim', fn) 
adata.obsm['consensus_estimate'].to_csv(fn)
adata.obsm['consensus_pvals'].to_csv(fn.replace('_estimate_', '_pvals_'))


# In[ ]:




