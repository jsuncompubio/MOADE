import anndata
import numpy as np
import pandas as pd
from .simulation import generate_simulated_data_omics1, generate_simulated_data_omics2
from .utils import ProcessInputData, process_purified_data
from .train import train_model, predict, reproducibility
from .model import AutoEncoder
from .jnmf import JointNMF



def Deconvolution(sc_rna, real_bulk1, real_bulk2, omics1='RNAseq', omics2=None,
                  d_prior=None, cell_type=None, subj_var=0.1, step_p=1e-3, step_s=1e-4, eps=1e-3, max_iter=500,
                  sparse=True, sparse_prob=0.5,
                  sep='\t', variance_threshold=0.98, scaler='mms', datatype='counts', genelenfile=None,
                  mode='high-resolution', adaptive=True, save_model_name=None,
                  batch_size=128, epochs=128, seed=0, output_dir=None):
    
    bulk_omic1 = pd.read_csv(real_bulk1, sep='\t', index_col=0)
    bulk_omic1 = bulk_omic1.transpose()
    bulk_omic1 = np.log(bulk_omic1 + 1)

    bulk_omic2 = pd.read_csv(real_bulk2, sep='\t', index_col=0)
    bulk_omic2 = bulk_omic2.transpose()
    if omics2 == 'Protein':
        bulk_omic2 = np.log(bulk_omic2)
    elif omics2 == 'ATACseq':
        bulk_omic2 = np.log(bulk_omic2 + 1)
    elif omics2 == 'DNAm':
        bulk_omic2 = -np.log(bulk_omic2)

    PropPred1, PropPred2, PurifiedSigm1, PurifiedSigm2, ini_prop = JointNMF(bulk_omic1, bulk_omic2, d_prior=d_prior, celltypes=cell_type, subj_var=subj_var, step_p=step_p, step_s=step_s, eps=eps, max_iter=max_iter, random_state=123)
    
    purified_bulk = [process_purified_data(PurifiedSigm2[cell_type[i]], omics=omics2) for i in range(len(cell_type))]

    simudata1, prop1 = generate_simulated_data_omics1(sc_data=sc_rna, d_prior=None, cell_type=cell_type, samplenum=5000, random_state=123, sparse=sparse, sparse_prob=sparse_prob)
    simudata2 = generate_simulated_data_omics2(pb_data=purified_bulk, cell_type=cell_type, prop_omic1=prop1, samplenum=5000, random_state=123, sparse=sparse, sparse_prob=sparse_prob, omics=omics2)

    train_x1, train_y1, test_x1, genename1, celltypes, samplename = \
        ProcessInputData(simudata1, real_bulk1, sep=sep, datatype=datatype, variance_threshold=variance_threshold,
                         scaler=scaler,
                         genelenfile=genelenfile, omics=omics1)
    train_x2, train_y2, test_x2, genename2, celltypes, samplename = \
        ProcessInputData(simudata2, real_bulk2, sep=sep, datatype='counts', variance_threshold=variance_threshold,
                         scaler=scaler,
                         genelenfile=None, omics=omics2)
    print('training data shape is ', train_x1.shape, train_x2.shape, '\ntest data shape is ', test_x1.shape, test_x2.shape)
    if save_model_name is not None:
        reproducibility(seed)
        model = train_model(train_x1, train_y1, train_x2, train_y2, save_model_name, batch_size=batch_size, epochs=epochs)
    else:
        reproducibility(seed)
        model = train_model(train_x1, train_y1, train_x2, train_y2, batch_size=batch_size, epochs=epochs)
    print('Notice that you are using parameters: mode=' + str(mode) + ' and adaptive=' + str(adaptive))
    if adaptive is True:
        if mode == 'high-resolution':
            CellTypeSigm1, TestPred1, CellTypeSigm2, TestPred2 = \
                predict(test_x1=test_x1, test_x2=test_x2, genename1=genename1, genename2=genename2, celltypes=celltypes, samplename=samplename,
                        model=model, model_name=save_model_name,
                        adaptive=adaptive, mode=mode)
            return CellTypeSigm1, TestPred1, CellTypeSigm2, TestPred2

        elif mode == 'overall':
            Sigm1, Pred1, Sigm2, Pred2 = \
                predict(test_x1=test_x1, test_x2=test_x2, genename1=genename1, genename2=genename2, celltypes=celltypes, samplename=samplename,
                        model=model, model_name=save_model_name,
                        adaptive=adaptive, mode=mode)
            return Sigm1, Pred1, Sigm2, Pred2
    else:
        Pred1, Pred2 = predict(test_x1=test_x1, test_x2=test_x2, genename1=genename1, genename2=genename2, celltypes=celltypes, samplename=samplename,
                       model=model, model_name=save_model_name,
                       adaptive=adaptive, mode=mode)
        Sigm1 = Sigm2 = None
        return Sigm1, Pred1, Sigm2, Pred2
