import json
from constants import DIALOGSUM, SAMSUM

def split_summary_into_sentences(summary):
    sentences = [sent.strip() + '.' for sent in summary.split('.') if len(sent.strip()) > 0]
    return sentences

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss    


def load_data(json_path, is_test, args):
    if args.dataset == SAMSUM:
        data = json.load(open(json_path,'r'))
    elif args.dataset == DIALOGSUM:
        data = [json.loads(l) for l in open(json_path,'r').readlines()]
        # gather multiple summaries
        if is_test:
            for sample in data:
                sample['summaries'] = [sample['summary1'], sample['summary2'], sample['summary3']]
    else:
        raise NotImplementedError

    return data

