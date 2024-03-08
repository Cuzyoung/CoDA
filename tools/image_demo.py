# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import os
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette

#/share/home/dq070/CoT/MIC-metawithdot/seg/meta_workdir/meta_2prompt_zeros_rfs2400_twiimd400_1200_night400_4000
#/share/home/dq070/CoT/MIC-origin/seg/work_dirs/cs2acdc/origin_new_conda_seed_1_4000eval
def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='/share/home/dq070/CoT/MIC-origin/seg/visual/rain/GOPR0572_frame_000830_rgb_anon.png')
    parser.add_argument('--config', help='Config file', default='/share/home/dq070/CoT/MIC-origin/seg/configs/mic/csHR2acdcHR_mic_hrda.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='/share/home/dq070/CoT/MIC-metawithdot/seg/meta_workdir/cs2acdc/new/Luna_Random_1/iter_60000.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    file, extension = os.path.splitext(args.img)
    pred_file = f'{file}_pred_CoDA2{extension}'
    assert pred_file != args.img
    model.show_result(
        args.img,
        result,
        palette=get_palette(args.palette),
        out_file=pred_file,
        show=False,
        opacity=args.opacity)
    print('Save prediction to', pred_file)


if __name__ == '__main__':
    main()
