"""Check dependencies/upstream hashes and optionally forward all three models."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Finite 64x64 forward for all models; no data/weights needed')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    report = {'python': sys.version.split()[0], 'dependencies': {}, 'upstream': {}, 'models': {}, 'failures': []}
    for name in ['torch', 'torchvision', 'torch_pruning', 'numpy', 'scipy', 'skimage', 'matplotlib', 'PIL', 'yaml', 'six', 'thop']:
        try:
            module = importlib.import_module(name)
            report['dependencies'][name] = getattr(module, '__version__', 'unknown')
        except Exception as error:
            report['failures'].append(name + ': ' + str(error))
    manifest = json.loads((ROOT / 'upstream_source_sha256.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError('Invalid upstream path: ' + relative)
        ok = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        report['upstream'][relative] = ok
        if not ok:
            report['failures'].append('upstream: ' + relative)
    if not report['failures']:
        from core.config import parse_args
        report['default_num_workers'] = parse_args([]).num_workers
        if report['default_num_workers'] != 0:
            report['failures'].append('Expected safe num_workers=0 default')
    if args.smoke and not report['failures']:
        import torch
        from core.modeling import build_model, extract_primary_prediction
        torch.set_num_threads(2)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu')
        for name in ['DNANet', 'UIUNet', 'ISNet']:
            try:
                model = build_model(name, mode='test').model.to(device).eval()
                with torch.inference_mode():
                    prediction = extract_primary_prediction(model(torch.zeros(1, 1, 64, 64, device=device)), name)
                ok = list(prediction.shape) == [1, 1, 64, 64] and bool(torch.isfinite(prediction).all())
                report['models'][name] = {'passed': ok, 'shape': list(prediction.shape)}
                if not ok:
                    report['failures'].append(name)
                del model, prediction
            except Exception as error:
                report['failures'].append(name + ': ' + str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
