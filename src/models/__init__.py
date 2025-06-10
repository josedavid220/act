from os import path as osp

from .act_net import ActTime
from .act_module import ACTLitModule


def create_model(args, is_train=False):
    if is_train:
        return ACTLitModule(net=ActTime(args), args=args)

    else: # test setting
        if args.ckpt_path is not None:
            # use pretrained parameter
            assert osp.exists(args.ckpt_path), print(f'checkpoint not exists: {args.ckpt_path}')
            print(f'Loading checkpoint from: {args.ckpt_path}')

            return ACTLitModule.load_from_checkpoint(args.ckpt_path, args=args, net=ActTime(args))
        
        else:
            raise ValueError('Need release option or checkpoint path')