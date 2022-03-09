import numpy as np
import pandas as pd
import anndata as ad
import torch
import scanpy as sc

# https://anndata-tutorials.readthedocs.io/en/latest/getting-started.html

n_obs, n_vars = 1000, 10
X = np.random.random((n_obs, n_vars))
# by convention X is a matrix that stores the observations in rows, the variable
# in columns. So each observation is like an experiment, a data point, and it
# has n_vars dimensions.

df = pd.DataFrame(X, columns=list("ABCDEFGHIJ"),
        index= np.arange(n_obs, dtype=int).astype(str),
        #index = ["obs " + i for i in np.arange(n_obs, dtype=int).astype(str)],
        )

# store the metadata about the data set in another dataframe
obs_meta = pd.DataFrame({
        'time_yr': np.random.choice([0, 2, 4, 8], n_obs),
        'subject_id': np.random.choice(['subject 1', 'subject 2', 'subject 4', 'subject 8'], n_obs),
        'instrument_type': np.random.choice(['type a', 'type b'], n_obs),
        'site': np.random.choice(['site x', 'site y'], n_obs),
    },
    index=np.arange(n_obs, dtype=int).astype(str),    # these are the same IDs of observations as above!
)

# joint both into an anndata object
adata = ad.AnnData(df, obs=obs_meta)
print(adata)

adata.to_df().head()

adata.obs.head()

# lets open Clemens' data (scrnaseq)
cdata = ad.read("./data/limb_sce_alternative.h5ad",)

print(cdata)

cdata.to_df().head()
# rows = obs = cell or whatever, with name indicating parameters such as mouse id and
# sex. columns = gene expression counts.
cdata.to_df().head().sum(axis=1)

cdata.obs.head()
# total column (or sum?) in obs is the total gene counts per observation.
cdata.obs[['sum', 'total']]

adata_subset = adata[:3, ['A', 'B']]
#adata.write('my_results.h5ad')

# layers is a dictionary-like object, with values of the same dimensions as X:
cdata.layers['logcounts']


# scanpy
# sc.pp.... contains preprocessing functions
# sc.tl... tools ant other type of transformation
# sc.pl .... plotting
# more: https://scanpy.readthedocs.io/en/stable/api.html#module-scanpy.pp

blobs = sc.datasets.blobs(n_variables=5, n_centers=5, cluster_std=1.0,
        n_observations=100)

print(blobs)
blobs.to_df().head()
blobs.obs

sc.pp.log1p(blobs)


X_normalized = sc.pp.normalize_total(cdata, target_sum=1,
        exclude_highly_expressed=False, key_added="normalization_factor",
        inplace=False,)

sc.tl.pca(blobs,)

print(blobs)
sc.pl.pca(blobs, color='blobs')

# the pc coordinates of the data
blobs.obsm["X_pca"]


## doing the full turoial ... ish
sc.settings.verbosity = 3
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

adata = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/',
        var_names='gene_symbols',)

sc.pl.highest_expr_genes(adata, n_top=15)

# filter out cells that don't express at least min_genes
sc.pp.filter_cells(adata, min_genes=200)
print(adata)

# filter out genes that are detected in less then min_cells
sc.pp.filter_genes(adata, min_cells=3)
print(adata)
adata.obs.head()
adata.var['n_cells']

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# filter out overly expressing cells etc.
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# notmalize, then logarithmize the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)

sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)

# pca
sc.tl.pca(adata)
sc.pl.pca(adata, color='MT-ND4')
sc.pl.pca_variance_ratio(adata, log=True)

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

sc.tl.umap(adata,)
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], )
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)
sc.tl.leiden(adata)

sc.pl.umap(adata, color=['leiden', 'CST3'])




### CDATA analysis
cdata = ad.read("./data/limb_sce_alternative.h5ad",)

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=160, facecolor='white')

cdata.var_names
cdata.var_names.str.startswith('Z').sum()

cdata.raw = cdata
cdata.X = cdata.layers["logcounts"]
sc.tl.pca(data=cdata)
sc.pl.pca(cdata, color='cell_ontology_id')
sc.pp.neighbors(cdata, n_neighbors=10, n_pcs=40)


sc.tl.umap(cdata, )

sc.pl.umap(cdata, color=['cell_ontology_id', 'cell_ontology_class', 'tissue',
    'subtissue'],)
