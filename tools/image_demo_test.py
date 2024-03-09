# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction
import argparse
import os
import argparse
import os
import sys
from argparse import ArgumentParser
from PIL import Image
import mmcv
#from pandocfilters import Image
from tqdm import tqdm

from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', help='Image dir',
                        default='') 
    parser.add_argument('--configs', help='Config file',
                        default='')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='')
    parser.add_argument('--out_dir', help='Output directory',
                        default='')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    # build the model from a configs file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.configs)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    if args.img_dir is not None:
        outdir = args.out_dir
        trainLabelIdDir = os.path.join(outdir, 'labelTrainIds')
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(trainLabelIdDir, exist_ok=True)
        for filename in tqdm(os.listdir(args.img_dir)):
            img = os.path.join(args.img_dir, filename)
            result = inference_segmentor(model, img)
            # print(result.shape)
            file, extension = os.path.splitext(filename)  #split the filename and extension name

            labelTrainId = Image.fromarray(result[0].astype('uint8')).convert('P')
            labelTrainId.save(os.path.join(trainLabelIdDir, f'{file}_gt_labelTrainIds.png'))
if __name__ == '__main__':
    main()

