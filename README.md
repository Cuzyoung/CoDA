## CoDA: Instructive Chain-of-Domain Adaptation with Severity-Aware Visual Prompt Tuning 

🌟🌟🌟 Here is the official project of :violin:[CoDA](). We only release the checkpoint for inference now and will release the code of CoDA.

🔥🔥🔥CoDA is a UDA methodology that boosts models to understand all adverse scenes (☁️,☔,❄️,&#x1F319;) by highlighting the discrepancies within these scenes.
CoDA achieves state-of-the-art performances on widely used benchmarks.

![night](images/demo1.png)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coda-instructive-chain-of-domain-adaptation/domain-adaptation-on-cityscapes-to)](https://paperswithcode.com/sota/domain-adaptation-on-cityscapes-to?p=coda-instructive-chain-of-domain-adaptation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coda-instructive-chain-of-domain-adaptation/domain-adaptation-on-cityscapes-to-1)](https://paperswithcode.com/sota/domain-adaptation-on-cityscapes-to-1?p=coda-instructive-chain-of-domain-adaptation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coda-instructive-chain-of-domain-adaptation/domain-adaptation-on-cityscapes-to-acdc)](https://paperswithcode.com/sota/domain-adaptation-on-cityscapes-to-acdc?p=coda-instructive-chain-of-domain-adaptation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coda-instructive-chain-of-domain-adaptation/semantic-segmentation-on-nighttime-driving)](https://paperswithcode.com/sota/semantic-segmentation-on-nighttime-driving?p=coda-instructive-chain-of-domain-adaptation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coda-instructive-chain-of-domain-adaptation/semantic-segmentation-on-dark-zurich)](https://paperswithcode.com/sota/semantic-segmentation-on-dark-zurich?p=coda-instructive-chain-of-domain-adaptation)
<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=Cuzyoung.CoDA&left_color=%23DFA3CB&right_color=%23CEE75F"> </a> 
<!-- 
 ![visitors](https://visitor-badge.glitch.me/badge?page_id=Cuzyoung.CoDA&left_color=%23DFA3CB&right_color=%23CEE75F) -->

![CoDA](images/Architec.png)

| Experiments | mIoU | Checkpoint | Configs |
|-|-|-|-|
|**Cityscapes $\rightarrow$ ACDC**|**72.6**|-|-|
|**Cityscapes $\rightarrow$ Foggy Zurich**|**60.9**|-|-|
|**Cityscapes $\rightarrow$ Foggy Driving**|**61.0**|-|-|
|**Cityscapes $\rightarrow$ Dark Zurich**|**61.2**|-|-|
|**Cityscapes $\rightarrow$ Nighttime Driving**|**59.2**|-|-|
|**Cityscapes $\rightarrow$ BDD100K-Night**|**41.6**|-|-|

## Download Checkpoint
```bash
cd CoDA
python ./tools/download_ck.py
```
or you can manually download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1NKfgJZtLGXpqs7zKvI8KpKpJmTYCRtyB?usp=drive_link).

## Environment
```
conda create -n coda python=3.8.5 pip=22.3.1
conda activate coda
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```
Before run demo, first configure the PYTHONPATH, or you may encounter error like 'can not found tools...'.
```bash
cd CoDA
export PYTHONPATH=.:$PYTHONPATH
```
or directly modify the .bashrc file
```bash
vi ~/.bashrc
export PYTHONPATH=your path/CoDA:$PYTHONPATH
source ~/.bashrc
```

## demo
```bash
python ./tools/image_demo.py --img ./images/night_demo.png --config ./configs/coda/csHR2acdcHR_coda.py --checkpoint ./pretrained/CoDA_cs2acdc.pth
```
## Inference Steps
```bash
python ./tools/image_demo.py --img_dir ./acdc_dir --config ./configs/coda/csHR2acdcHR_coda.py --checkpoint ./pretrained/CoDA_cs2acdc.pth --out_dir ./workdir/cs2acdc
```
## Traning Steps
```bash
python ./tools/train.py --config ./configs/coda/csHR2acdcHR_coda.py --work-dir ./workdir/cs2acdc
```
