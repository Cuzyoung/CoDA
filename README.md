## CoDA: Instructive Chain-of-Domain Adaptation with Severity-Aware Visual Prompt Tuning 


🌟💫 Here is the official project of [CoDA](). 

🔥🔥🔥CoDA is a UDA methodology that boosts models to understand all adverse scenes by highlighting the discrepancies within these scenes.
CoDA achieves state-of-the-art performances on widely used benchmarks.

![SYSU](https://img.shields.io/badge/SYSU-095101)&nbsp;![NUS](https://img.shields.io/badge/NUS-003D7C)&nbsp;![EPFL](https://img.shields.io/badge/EPFL-F60000)&nbsp;<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=Cuzyoung.CoDA&left_color=%23DFA3CB&right_color=%23CEE75F"> </a> 
![CoDA](images/Architec.png)



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