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
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--img_dir', help='Image dir',
                        default='/share/home/dq070/hy-tmp/AS_id_all/NighttimeDriving/img/val') #/share/home/dq070/hy-tmp/daformer_demodata/fog/img/test/  /share/home/dq070/lfh/fog_medium/fog/img/val  
    # /share/home/dq070/hy-tmp/AS_id_all/fog-driving/img2/share/home/dq070/lfh/dark-zurich/img/share/home/dq070/hy-tmp/AS_id_all/NighttimeDriving/img/val
    parser.add_argument('--configs', help='Config file',
                        default='/share/home/dq070/CoT/MIC-metawithdot/seg/configs/mic/' 
                                'csHR2bddHR_mic_hrda.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='/share/home/dq070/CoT/MIC-metawithdot/seg/meta_workdir/cs2dz/CoDA_0.05_1/'
                                'iter_44000.pth')
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
    # # test a single image
    # result = inference_segmentor(model, args.img)
    # # show the results
    # file, extension = os.path.splitext(args.img)
    # pred_file = f'{file}_pred{extension}'
    # assert pred_file != args.img
    # model.show_result(
    #     args.img,
    #     result,
    #     palette=get_palette(args.palette),
    #     out_file=pred_file,
    #     show=False,
    #     opacity=args.opacity)
    # print('Save prediction to', pred_file)
    if args.img_dir is not None:
        outdir = '/share/home/dq070/CoT/MIC-metawithdot/seg/meta_workdir/cs2dz/CoDA_0.05_1/results/nd'
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

