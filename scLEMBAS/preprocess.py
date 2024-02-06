"""
Utility functions for single-cell AnnData objects.
Functions to process single-cell AnnData objects.
"""

import decoupler as dc
from anndata import AnnData
from decoupler.pre import extract

def get_tf_activity(adata, organism: str, grn = 'collectri', 
                    verbose: bool = True, min_n: int = 5, use_raw: bool = False,
                    **kwargs):
    """Wrapper of decoupler to estimate TF activity from single-cell transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell data matrix 
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name, latin name or Ensembl name. 
        Organisms other than human will be translated from human data by orthology.
    grn : str, optional
        database to get the GRN, by default 'collectri'. Available options are ``collectri`` or ``dorothea``.
    min_n : int
        Minimum of targets per source. If less, sources are removed. By default 5.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use AnnData its raw attribute.
    kwargs : 
        passed to  `decoupler.decouple`.

    Returns
    -------
    estimate : DataFrame
        Consensus TF activity scores. Stored in `.obsm['consensus_estimate']`.
    pvals : DataFrame
        Obtained TF activity p-values. Stored in `.obsm['consensus_pvals']`.
    """

    grn_map = {'collectri': dc.get_collectri, 'dorothea': dc.get_dorothea} # get_dorothea returns "A" confidence by default
    net = grn_map[grn](organism=organism, split_complexes=False) # builds on dorothea, used by Saez-Rodriguez lab

    # # reimplementation of dc.run_consensus, allowing all options in dc.decouple to be passed
    # dc.run_consensus(mat=adata, net=net, source='source', target='target', weight='weight', **kwargs)

    # m, r, c = extract(adata, use_raw=use_raw, verbose=verbose)
    if verbose:
        print('Running consensus.')
    
    # # unnecessary, this is the default behavior    
    # if not kwargs:
    #     kwargs = {'methods': ['lm', 'ulm', 'wsum'], 
    #               'cns_metds': ['lm', 'ulm', 'wsum_norm']}
    # else:
    #     if 'methods' not in kwargs:
    #         kwargs['methods'] = ['lm', 'ulm', 'wsum']
    #     if 'cns_methods' not in kwargs and kwargs['methods'] == ['lm', 'ulm', 'wsum']:
    #         kwargs['cns_metds'] = ['lm', 'ulm', 'wsum_norm']

    res = dc.decouple(mat=adata, net=net, source='source', target='target', weight='weight', consensus = True,
                      min_n=min_n, verbose=verbose, use_raw=use_raw, **kwargs)