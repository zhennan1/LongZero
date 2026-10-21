import torch,json,os

def sample(optimizer):
    chunks=[]
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None or p.numel()==0: continue
            q=p.detach()
            if hasattr(q,'to_local'): q=q.to_local()
            q=q.reshape(-1)
            if q.numel(): chunks.append(q[::max(1,q.numel()//128)][:128].float().clone())
    return torch.cat(chunks) if chunks else torch.empty(0)

def before(engine,grad_norm):
    if not torch.isfinite(grad_norm).all():
        raise RuntimeError('LCZERO_NONFINITE_GRAD: refusing silent skipped update')
    engine._lczero_update_count=getattr(engine,'_lczero_update_count',0)+1
    if engine._lczero_update_count<=16 or engine._lczero_update_count%72==0:
        return sample(engine.optimizer)
    return None

def after(engine,old):
    if old is None: return
    new=sample(engine.optimizer); delta=new-old
    finite=bool(torch.isfinite(new).all().item())
    changed=int(torch.count_nonzero(delta).item())
    print('LCZERO_UPDATE_AUDIT '+json.dumps({'rank':torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,'optimizer_step':engine._lczero_update_count,'sampled':old.numel(),'changed':changed,'max_abs_delta':float(delta.abs().max().item()) if delta.numel() else 0,'finite':finite}),flush=True)
    if not finite: raise RuntimeError('LCZERO_NONFINITE_WEIGHT')
