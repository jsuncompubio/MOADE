import numpy as np
import pandas as pd
 

# Joint-NMF
def grad_p(X1, Y1, X2, Y2, p, p_init, s1, s2):
    grad = np.mean(np.transpose((np.dot(X1, p * s1) - Y1) * X1) * s1, axis=1).reshape(-1, 1) + np.mean(np.transpose((np.dot(X2, p * s2) - Y2) * X2) * s2, axis=1).reshape(-1, 1) + (p - p_init)
    return grad

def grad_si(X, Y, p, s):
    grad = np.mean(np.transpose((np.dot(X, p * s) - Y) * X) * p, axis=1).reshape(-1, 1)
    return grad

def grad_Xi(X, Y, p, s):
    grad = np.dot(np.dot(X, p * s) - Y, np.transpose(p * s))
    return grad

def jnmf_optimizer(Y1, Y2, ini_p, ini_s, X1, X2, step_p, step_s, eps, max_iter):
    p = ini_p.reshape(-1, 1).copy()  
    pt = ini_p.reshape(-1, 1).copy() 
    s1 = ini_s[0].reshape(-1, 1)
    s2 = ini_s[1].reshape(-1, 1)
    eps_t = 0.2
    iter_t = 1
    X1_init = X1.copy()
    X2_init = X2.copy()
    res = {}
            
    while eps_t > eps and iter_t < max_iter:
        # update parameters p, s1 and s2 based on gradients
        p_u = np.maximum(p - step_p * grad_p(X1, Y1, X2, Y2, p, pt, s1, s2), 0)
        s1_u = np.maximum(s1 - step_s * grad_si(X1, Y1, p, s1), 0)
        s2_u = np.maximum(s2 - step_s * grad_si(X2, Y2, p, s2), 0)
        
        # update sigmatrix X1 and X2
        X1_u = X1 - 0.1 * grad_Xi(X1, Y1, p, s1)
        X2_u = X2 - 0.1 * grad_Xi(X2, Y2, p, s2)
        
        # adjust negative values in updated sigmatrix
        if np.min(Y1) >= 0:
            X1_u[X1_u < 0] = 0
        if np.min(Y2) >= 0:
            X2_u[X2_u < 0] = 0
        
        # update convergence threshold eps_t
        eps_t = max(np.max(np.abs(p_u - p)),
                    np.max(np.abs(p * s1 - p_u * s1_u)),
                    np.max(np.abs(p * s2 - p_u * s2_u)),
                    np.max(np.abs(X1_u - X1)),
                    np.max(np.abs(X2_u - X2)))
        
        # update variables for next iteration
        p, s1, s2 = p_u, s1_u, s2_u
        max_diff_X1 = np.max(np.abs(X1_u - X1))
        max_diff_X2 = np.max(np.abs(X2_u - X2))
        X1, X2 = X1_u, X2_u
        iter_t += 1
    
    if iter_t >= max_iter and eps_t > eps:
        print('Max iteration reached without convergence')
    
    print(f'iter = {iter_t} eps = {eps_t}\n')
    
    # store results in dictionary
    res['iter'] = iter_t
    res['L'] = np.sum(np.square(np.dot(X1, p * s1) - Y1)) + np.sum(np.square(np.dot(X2, p * s2) - Y2))
    res['p'] = p
    res['s1'] = s1
    res['s2'] = s2
    res['eps'] = eps_t
    res['X1'] = X1
    res['X2'] = X2
    res['max_diff_X1'] = max_diff_X1
    res['max_diff_X2'] = max_diff_X2
    res['prop1'] = np.transpose(p * s1) / np.sum(p * s1)
    res['prop2'] = np.transpose(p * s2) / np.sum(p * s2)
    res['ini_p'] = ini_p
    res['X1_init'] = X1_init
    res['X2_init'] = X2_init
    
    return res


def JointNMF(bulk_omic1, bulk_omic2, d_prior, celltypes, subj_var = 0.1, step_p = 1e-3, step_s = 1e-3, eps = 1e-3, max_iter = 500, random_state=None):
	'''
	bulk_omic1: feature * sample dataframe of bulk input omic1
	bulk_omic2: feature * sample dataframe of bulk input omic2
	d_prior: estimated from cell counts
	celltypes: list of cell types
	'''
	feature_omic1 = bulk_omic1.index.tolist()
	feature_omic2 = bulk_omic2.index.tolist()
	samplename = bulk_omic1.columns.tolist()
	n_feature1 = len(feature_omic1)
	n_feature2 = len(feature_omic2)
	n_sample = bulk_omic1.shape[1]
	K = len(d_prior)

	# generate random initial prop
	alpha = (np.array(d_prior) * ((1 - subj_var) / subj_var)) / np.sum(np.array(d_prior))
	if isinstance(random_state, int):
		np.random.seed(random_state)
	ini_prop = np.random.dirichlet(alpha, n_sample)    # n_sample * K

	TestSigmList1 = np.zeros((n_sample, K, n_feature1))
	TestSigmList2 = np.zeros((n_sample, K, n_feature2))
	TestPred1 = np.zeros((n_sample, K))
	TestPred2 = np.zeros((n_sample, K))

	for i in range(n_sample):
		X1 = np.random.normal(loc=np.mean(bulk_omic1.to_numpy()), scale=np.std(bulk_omic1.to_numpy()), size=(n_feature1, K))
		X2 = np.random.normal(loc=np.mean(bulk_omic2.to_numpy()), scale=np.std(bulk_omic2.to_numpy()), size=(n_feature2, K))

		Y1 = bulk_omic1.iloc[:, i].to_numpy().reshape(-1, 1)
		Y2 = bulk_omic2.iloc[:, i].to_numpy().reshape(-1, 1)

		ini_p = ini_prop[i]
		ini_s = [np.ones(K), np.ones(K)]

		res = jnmf_optimizer(Y1, Y2, ini_p, ini_s, X1, X2, step_p, step_s, eps, max_iter)

		TestPred1[i, :] = res['prop1']
		TestPred2[i, :] = res['prop2']
		TestSigmList1[i, :, :] = np.transpose(res['X1'])
		TestSigmList2[i, :, :] = np.transpose(res['X2'])

	TestPred1 = pd.DataFrame(TestPred1, columns=celltypes, index=samplename)
	TestPred2 = pd.DataFrame(TestPred2, columns=celltypes, index=samplename)
	CellTypeSigm1, CellTypeSigm2 = {}, {}
	for i in range(len(celltypes)):
		cellname = celltypes[i]
		sigm = TestSigmList1[:, i, :]
		sigm = pd.DataFrame(sigm, columns=feature_omic1, index=samplename)
		CellTypeSigm1[cellname] = sigm
	for i in range(len(celltypes)):
		cellname = celltypes[i]
		sigm = TestSigmList2[:, i, :]
		sigm = pd.DataFrame(sigm, columns=feature_omic2, index=samplename)
		CellTypeSigm2[cellname] = sigm

	return TestPred1, TestPred2, CellTypeSigm1, CellTypeSigm2, ini_prop
