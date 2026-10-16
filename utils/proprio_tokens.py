"""Reuse frozen VLA state representations without training another state encoder."""
import re
import torch
from torch import Tensor


def pool_state_spans(hidden: Tensor, offsets: Tensor, spans: list, valid: Tensor) -> Tensor:
    """Align variable digit subtokens to stable state-coordinate positions."""
    count=len(spans[0])
    if not count or any(len(row)!=count for row in spans):
        raise ValueError('State coordinate count must be nonzero and constant within a batch')
    weights=torch.zeros(hidden.shape[0],count,hidden.shape[1],device=hidden.device,dtype=hidden.dtype)
    offsets=offsets.to(hidden.device)
    for b,row in enumerate(spans):
        for p,(start,end) in enumerate(row):
            selected=(offsets[b,:,1]>start)&(offsets[b,:,0]<end)&valid[b].to(hidden.device)
            if not selected.any():
                raise ValueError('State prompt was truncated or a state coordinate has no encoded tokens')
            weights[b,p]=selected.to(hidden.dtype)/selected.sum()
    return weights @ hidden


def pi05_state_tokens(hidden: Tensor, input_ids: Tensor, language_mask: Tensor,
                      prompts: list[str], tokenizer, *, state_dim: int) -> Tensor:
    if prompts is None:
        raise ValueError('PI05 state-token reuse requires the processed Task/State prompt')
    encoded=tokenizer(prompts,return_offsets_mapping=True,return_tensors='pt',
                      padding='max_length',max_length=input_ids.shape[1],truncation=True,padding_side='right')
    if not torch.equal(encoded['input_ids'],input_ids.cpu()):
        raise ValueError('PI05 state offsets do not match policy tokenizer input IDs')
    all_spans=[]
    for prompt in prompts:
        start=prompt.rfind(', State: ')
        end=prompt.rfind(';\nAction:')
        if start<0 or end<=start:
            raise ValueError('PI05 prompt has no complete State section')
        start+=len(', State: ')
        spans=[(start+m.start(),start+m.end()) for m in re.finditer(r'-?\d+',prompt[start:end])]
        if len(spans)!=state_dim:
            raise ValueError(f'PI05 State section has {len(spans)} coordinates, expected {state_dim}')
        all_spans.append(spans)
    language_hidden=hidden[:,-input_ids.shape[1]:].float()
    return pool_state_spans(language_hidden,encoded['offset_mapping'],all_spans,language_mask)


def xvla_state_component(projector, state: Tensor, domain_id: Tensor, *, action_dim: int) -> Tensor:
    """Extract s @ W_state from the existing domain-conditioned joint linear map.

    The joint bias belongs to the full action/state/time projection, so it is
    intentionally excluded. No weights are copied or added to the ACE model.
    """
    weight=projector.fc(domain_id).view(state.shape[0],projector.input_size,projector.output_size)
    state_weight=weight[:,action_dim:action_dim+state.shape[-1]]
    return torch.bmm(state[:,None].to(weight.dtype),state_weight).float()
