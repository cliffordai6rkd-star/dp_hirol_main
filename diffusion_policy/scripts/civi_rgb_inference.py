"""Apply an exported CIVI RGB model to an actual recorded causal window.

python -m diffusion_policy.scripts.civi_rgb_inference --model RUN/civi_rgb.pt \
    --input causal_window.pt --output enhanced.pt --device cuda:0
Input: {'images': {'side': [B,2,3,224,224], 'wrist': ...},
        'tau_ext': [B,2,8,7]}; RGB float32 in [0,1], torque in physical units.
"""
import argparse
import torch
from diffusion_policy.policy.civi_rgb import CiviRGB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    model = CiviRGB.load(args.model,args.device)
    window = torch.load(args.input,map_location=args.device,weights_only=False)
    result = model(window['images'],window['tau_ext'])
    torch.save({k:v.cpu() for k,v in result.items()},args.output)


if __name__ == '__main__':
    main()
