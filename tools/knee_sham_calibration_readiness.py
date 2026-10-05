"""One fictitious-RGB CPU MobileNet check of native/sham search reuse.

No pretrained cache, actual dataset, targets or scientific seed is accessed.
"""
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large

from tralo.knee_end_to_end import infer
from tralo.knee_yuval import transforms_for
from tralo.targeted_step import targeted_step


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only fixture entered CUDA')
    torch.set_num_threads(1)
    torch.manual_seed(8080003)  # Infrastructure-only, fictitious inputs.
    started = time.monotonic()
    model = mobilenet_v3_large(weights=None, num_classes=5).cpu()
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.zero_()
        model.classifier[-1].bias[3] = .01
    evaluation = transforms_for()[1]
    pool = [torch.stack([evaluation(Image.new('RGB', (230, 240), (31*i, 17*i, 11*i)))
                         for i in range(start, start+4)]) for start in [0, 4]]
    origin_hash = state_hash(model)
    native, legacy, reused = [copy.deepcopy(model) for _ in range(3)]
    calibration = []
    caps = [None, None, None, 2, None]  # Fictitious fixture cap, not the approved cohort.
    real = targeted_step(native, pool, caps, calibration_out=calibration)
    generator1, generator2 = [torch.Generator().manual_seed(8080010) for _ in range(2)]
    ordinary_rng = torch.get_rng_state().clone()
    old = targeted_step(legacy, pool, caps, sham_generator=generator1)
    new = targeted_step(reused, pool, caps, sham_generator=generator2, calibration=calibration[0])
    probabilities = {name:infer(value, pool) for name,value in [('native',native),('legacy_sham',legacy),('reused_sham',reused)]}
    if not all(torch.equal(value, reused.state_dict()[key]) for key,value in legacy.state_dict().items()):
        raise RuntimeError('MobileNet sham state differs')
    if not torch.equal(probabilities['legacy_sham'], probabilities['reused_sham']):
        raise RuntimeError('MobileNet sham probabilities differ')
    if not torch.equal(generator1.get_state(), generator2.get_state()) or not torch.equal(torch.get_rng_state(), ordinary_rng):
        raise RuntimeError('MobileNet sham RNG differs')
    if not real['applied'] or real['radius']!=old['radius'] or new['radius']!=old['radius']:
        raise RuntimeError('Fictitious matched radius differs or inactive')
    if new['evaluations']>=old['evaluations'] or state_hash(model)!=origin_hash or torch.cuda.is_initialized():
        raise RuntimeError('Search reduction or CPU/PTO boundary failed')
    paths={}
    for name,values in probabilities.items():
        data=json.dumps(values.tolist(),allow_nan=False).encode()
        path=output/(name+'.json');path.write_bytes(data)
        paths[path.name]=hashlib.sha256(data).hexdigest()
    receipt=dict(status='passed',validation_seed=8080003,scientific_seed_claims=0,
                 fictitious_RGB_images=8,pretrained_weights_used=False,actual_data_labels_or_scores_read=False,
                 model_state_entries=len(model.state_dict()),legacy_and_reused_state_sha256=state_hash(legacy),
                 original_state_sha256=origin_hash,all_state_and_probability_entries_equal=True,
                 seeded_generator_state_equal=True,ordinary_rng_unchanged=True,
                 native=real,legacy_sham=old,reused_sham=new,files=paths,
                 elapsed_cpu_wall_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 cuda_initialized=False,gpu_hours=0,
                 limitation='Fictitious-RGB unpretrained CPU MobileNet parity only. No real full-cohort dose, pretrained integration, CUDA parity/cost, scientific quality or campaign/budget certification follows.')
    (output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['native','legacy_sham','reused_sham','files']}))


if __name__=='__main__':
    main(sys.argv[1])
