import torch
import random
import torch.nn.functional as F
def frame_level_contrastive_loss(latent_vec, mask, K, temperature = 0.1):
        norm_vec = F.normalize(latent_vec, dim = -1)
        padding_mask = ~padding_mask
        mask = padding_mask[:,:-1] * padding_mask[:,1:]
        anchor = norm_vec[:,:-1,:]
        pos = norm_vec[:,1:,:] #Positive examples for the anchor
        anchor = anchor[mask.bool()]
        pos = pos[mask.bool()]
        sim_ap = torch.sum(anchor * pos, dim = -1) / temperature
        neg_pairs = []
        valid_neg = norm_vec[padding_mask.bool()]
        M, _ = valid_neg.shape 
        unpad_regime_len = torch.sum(padding_mask, dim = -1)
        unpad_regime_len_map = unpad_regime_len.repeat_interleave(unpad_regime_len)
        count = 0
        count_max = 0
        i  = 0
        while count_max < M:
            if i == unpad_regime_len_map[count + i] - 1:
                count += unpad_regime_len_map[count + i]
                i = 0
                count_max += 1
                continue
            neg_idx = list(range(count,count + i - 1)) + list(range(min(count + i + 2, count + unpad_regime_len_map[count + i] - 1),count + unpad_regime_len_map[count + i]))
            neg_idx_k = torch.tensor(random.sample(neg_idx, K - 1) + [count + i + 1])
            neg_pairs.append(valid_neg[neg_idx_k,:])
            count_max += 1
            i += 1
        neg_pairs = torch.stack(neg_pairs)
        sim_an = torch.sum(anchor.unsqueeze(1) * neg_pairs, dim = -1) / temperature
        logits = torch.cat([sim_ap.unsqueeze(1), sim_an], dim = 1).to(latent_vec.device)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=latent_vec.device)

        return F.cross_entropy(logits, labels) #Done