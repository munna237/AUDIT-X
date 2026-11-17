import json, sys, numpy as np, sklearn
sys.path.insert(0,'/tmp/regen'); import ev4, ev5
from fb import FB
exec(open('gen.py').read().split("KEYS=")[0].split("C={",1)[0])  # imports only
C=eval("{"+open('gen.py').read().split("C={",1)[1].split("\nKEYS=")[0])
KEYS={"Malimg":"/home/claude/keys/auditx_v3_group_preview/Malimg_image_keys.json","MaleVis":"/home/claude/realmv/image_keys.json","Dumpware10":"/home/claude/keys/auditx_v3_group_preview/Dumpware10_image_keys.json"}
ds=sys.argv[1]; ev=ev5 if ds=='MaleVis' else ev4; cfg=dict(ev.CFG); b=FB(C[ds]); kt=(0,1,2) if ds=='MaleVis' else (0,1)
keys=json.load(open(KEYS[ds]))['keys']
G=ev.compute_groups(b,KEYS[ds],kt,cfg['UNSPLITTABLE_SHARE'],cfg['MIN_CLASS_IMAGES'],cfg['NEAR_DUP_BITS'],cfg['NEAR_DUP_CAP'])
Gn=dict(G, groups=np.arange(len(b.samples)))   # naive: every image its own group; same exclusions and drops
fmd=np.array([k[0] for k in keys]); pmd=np.array([k[1] for k in keys]); dh=np.array([k[2] for k in keys])
bits=np.unpackbits(np.frombuffer(bytes.fromhex(''.join(dh)),dtype=np.uint8).reshape(len(dh),32),axis=1)
tg=np.array(b.targets)
def leak(sp):
    tr=np.asarray(sp['train']); ev_=np.concatenate([sp['calib'],sp['test']]).astype(int)
    exact=np.isin(fmd[ev_],fmd[tr])|np.isin(pmd[ev_],pmd[tr]); same256=np.isin(dh[ev_],dh[tr])
    near=np.zeros(len(ev_),bool)
    for c in np.unique(tg[ev_]):                      # within-family Hamming <= 8, as in the protocol
        e=ev_[tg[ev_]==c]; t=tr[tg[tr]==c]
        if len(t)==0: continue
        d=(bits[e][:,None,:]!=bits[t][None,:,:]).sum(2) if len(e)*len(t)<4e6 else None
        if d is None:
            m=np.zeros(len(e),bool)
            for i0 in range(0,len(e),200): m[i0:i0+200]=((bits[e[i0:i0+200]][:,None,:]!=bits[t][None,:,:]).sum(2)<=8).any(1)
        else: m=(d<=8).any(1)
        near[tg[ev_]==c]=m
    return 100*exact.mean(),100*same256.mean(),100*near.mean()
res={'grouped':[],'naive':[]}; fams=[]
for s in range(5):
    g=ev.make_split(b,s,cfg,G); n=ev.make_split(b,s,cfg,Gn)
    assert g['unknown_names']==n['unknown_names'] and sorted(g['unk'])==sorted(n['unk'])
    fams.append(g['unknown_names']); res['grouped'].append(leak(g)); res['naive'].append(leak(n))
out={k:np.mean(v,0).round(2).tolist() for k,v in res.items()}
print('RESULT',ds,'identical unknown families in all 5 splits; leak % (exact file/pixel, identical 256-bit, within 8 bits):',out)
json.dump(dict(dataset=ds,unknown_families=fams,**out,per_split=res),open(f'/tmp/regen/naive_leak_{ds}.json','w'))
