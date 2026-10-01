from math import ceil
def gemm(M,N,K,T):
    m,n,k=ceil(M/T),ceil(N/T),ceil(K/T)
    return m*n*k, m*k+k*n+m*n   # compute tiles, dma tiles (A,B in, C out)
def layer_events(tok,d,heads,hd,kvh,ffn,kv_len,T,gated=True,nexp=1):
    ev=[]
    ev.append(gemm(tok,(heads+2*kvh)*hd,d,T))          # QKV
    for _ in range(heads):
        ev.append(gemm(tok,kv_len,hd,T))               # QK^T
        ev.append(gemm(tok,hd,kv_len,T))               # PV
    ev.append(gemm(tok,d,heads*hd,T))                  # O proj
    for _ in range(nexp):
        ev.append(gemm(tok,ffn,d,T))                   # up
        if gated: ev.append(gemm(tok,ffn,d,T))         # gate
        ev.append(gemm(tok,d,ffn,T))                   # down
    c=sum(e[0] for e in ev); dm=sum(e[1] for e in ev)
    return c,dm
def total(T):
    out={}
    # SigLIP So400m: 27 layers, d1152, 16 heads x72, FFN 4304 (not gated), 3 images x 256 tokens
    c,dm=layer_events(256,1152,16,72,16,4304,256,T,gated=False)
    out['vision']=(c*27*3,dm*27*3)
    # PaliGemma 2B: 18 layers, d2048, 8 heads x256, MQA kv1, ffn 16384 gated, 816 tokens
    c,dm=layer_events(816,2048,8,256,1,16384,816,T)
    out['vlm_prefill']=(c*18,dm*18)
    # action expert: 18 layers, d1024, 8x256, kv1, ffn 4096 gated, 51 tokens attend over 867; MoE 4 routed+1 shared, all 4 live -> 5 FFNs, 10 steps
    c,dm=layer_events(51,1024,8,256,1,4096,867,T,nexp=5)
    out['action_x10']=(c*18*10,dm*18*10)
    return out
for T in (32,64,128,256):
    o=total(T); C=sum(v[0] for v in o.values()); D=sum(v[1] for v in o.values())
    print(f"T={T:4d}  compute={C:>9,}  dma={D:>9,}  total={C+D:>9,}   "+"  ".join(f"{k}={v[0]+v[1]:,}" for k,v in o.items()))
