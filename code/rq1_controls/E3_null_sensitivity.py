import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
R=pd.read_csv('/home/claude/repo/AUDIT-X/results/primary_analysis/runs_long.csv')
M=['agef_sharp','agef_tuned','avg_joint','deep_ensemble','single_mobilenet','single_resnet18','single_effnet_b0']; S=['msp','entropy','maxlogit','energy','knn50','mahalanobis']
P=R[R.model.isin(M)&R.score.isin(S)]; rng=np.random.RandomState(0); B=3000; rows=[]
for ds in ['Malimg','MaleVis','Dumpware10']:
    g=P[P.dataset==ds]; sp=sorted(g.split.unique())
    X={s:np.stack([np.stack([g[(g.split==s)&(g.model==m)&(g.score==k)].sort_values('seed').zdr5.values for k in S]) for m in M]) for s in sp}
    E={s:(X[s]-X[s].mean(2,keepdims=True))*np.sqrt(3/2) for s in sp}
    sd=np.array([np.sqrt(np.mean(np.concatenate([E[s][m].ravel()**2 for s in sp]))) for m in range(7)])
    cov=[np.cov(np.concatenate([E[s][m].T for s in sp],axis=0),rowvar=False) for m in range(7)]
    obs=np.mean([X[s].mean(2).max()-X[s].mean(2).mean() for s in sp])
    res={'blocked (equal noise)':np.zeros(B),'rescaled blocks':np.zeros(B),'model-specific parametric':np.zeros(B)}
    src=np.repeat(np.arange(7),3); slot=np.repeat(np.arange(7),3)
    for s in sp:
        mu=X[s].mean(2).mean(); blk=E[s].transpose(0,2,1).reshape(21,6)
        for b in range(B):
            pm=rng.permutation(21); Yb=blk[pm]
            res['blocked (equal noise)'][b]+=(mu+Yb.reshape(7,3,6).mean(1)).max()-mu
            Yr=Yb*(sd[slot]/np.maximum(sd[src[pm]],1e-9))[:,None]
            res['rescaled blocks'][b]+=(mu+Yr.reshape(7,3,6).mean(1)).max()-mu
            Yp=np.stack([rng.multivariate_normal(np.zeros(6),cov[m]/3,check_valid='ignore') for m in range(7)])
            res['model-specific parametric'][b]+=(mu+Yp).max()-mu
    for k,v in res.items():
        v=v/len(sp); rows.append(dict(dataset=ds,null=k,observed_pp=100*obs,null_mean_pp=100*v.mean(),real_pp=100*(obs-v.mean()),real_share=(obs-v.mean())/obs,p=(1+np.sum(v>=obs))/(B+1)))
    print(ds,'single-run SD of UFR by model (pp):',dict(zip(['GE','GE-T','Avg','DE','MNv3','RN18','EffB0'],np.round(100*sd,1))),flush=True)
D=pd.DataFrame(rows); print(D.round(3).to_string(index=False)); D.to_csv('/tmp/E3_null_sensitivity.csv',index=False)
