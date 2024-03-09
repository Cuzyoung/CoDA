## CoDA: Instructive Chain-of-Domain Adaptation with Severity-Aware Visual Prompt Tuning 
<font face="Calibri">

🌟🌟🌟 Here is the official project of :violin:[CoDA](). 

🔥🔥🔥CoDA is a UDA methodology that boosts models to understand all adverse scenes (☁️,☔,❄️,&#x1F319;) by highlighting the discrepancies within these scenes.
CoDA achieves state-of-the-art performances on widely used benchmarks.

![SYSU](https://img.shields.io/badge/SYSU¹-095101)&nbsp;![WUST](https://img.shields.io/badge/WUST-95C4D6)&nbsp;![NUS](https://img.shields.io/badge/NUS-003D7C)&nbsp;![EPFL](https://img.shields.io/badge/EPFL-F60000)&nbsp;![Oxford*](https://img.shields.io/badge/Oxford*-F1C86C)&nbsp;<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=Cuzyoung.CoDA&left_color=%23DFA3CB&right_color=%23CEE75F"> </a> 
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

## Traning Steps
```bash
Python ./tools/train.py --config ./configs/coda/csHR2acdcHR_coda.py --work-dir ./workdir/cs2acdc
```
## demo
```bash
Python ./tools/image_demo.py --img ./images/night_demo.png --config ./configs/coda/csHR2acdcHR_coda.py --checkpoint ./pretrained/CoDA_cs2acdc.pth
```
## Inference Steps
```bash
Python ./tools/image_demo.py --img_dir ./acdc_dir --config ./configs/coda/csHR2acdcHR_coda.py --checkpoint ./pretrained/CoDA_cs2acdc.pth --out_dir ./workdir/cs2acdc
```
</font>