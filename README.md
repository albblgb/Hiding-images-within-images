# pusnet
This repository is the official code for pusnet and the reproduced code for balujanet, hidden, wengnet and hinet.

* [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html) 
  * [*Guobiao Li*](https://tomtomtommi.github.io/), [*Sheng Li*](http://www.commsp.ee.ic.ac.uk/~xindeng/), [*Zicong Luo*](http://shi.buaa.edu.cn/MaiXu/zh_CN/index.htm), [*Zhengxing Qian*](http://buaamc2.net/html/Members/jianyiwang.html), [*XinPeng Zhang*](http://cst.buaa.edu.cn/info/1071/2542.htm).


Published on [**ACM MM 2023**](http://iccv2021.thecvf.com/home).
By [MAS Lab](http://buaamc2.net/) @ [Fudan University](http://ev.buaa.edu.cn/).

<center>
  <img src=https://github.com/TomTomTommi/HiNet/blob/main/HiNet.png width=60% />
</center>
 
## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:
    `conda env create  -f env.yml`
    `conda activate NIPS`


## Get Started
- If train, change the code in `config.py`

    `line4:  mode = 'train' ` 

- Then run `python *net.py`, for example, `python wengnet.py`


- If test (Here we provide [trained models](https://drive.google.com/drive/folders/1lM9ED7uzWYeznXSWKg4mgf7Xc7wjjm8Q?usp=sharing).), change the code in `config.py`

    `line4:  mode = 'test' ` 
    `line36-41:  test_*net_path = '' `

- Then run `python *net.py`


- The processed images, such as stego image and recovered secret image, will be saved at 'results/images'
- The training or testing log will be saved at 'results/*.log'


## Dataset
- In this paper, we use the commonly used dataset DIV2K for trainging, DIV2K, COCO, and ImageNet for testing.

- image folder

- For train or test on the dataset,  e.g.  DIV2K, change the code in `config.py`:

    `line17:  data_dir = '' ` 
    `data_name_train = 'div2k'`
    `data_name_test = 'div2k'`
    `line30:  suffix = 'png' `

- Structure of the dataset directory:
    data_dir 
    |--- div2k
        |--- train
            |--- 0001.png
            |--- 0002.png
            |...
        |--- test
            |--- 0001.png
            |--- 0002.png
            |...
    |--- coco
        |...
    |--- imagenet
        |...

    

## Others
- The `batch_size` in `config.py` should be at least `2*number of gpus` and it should be divisible by number of gpus.

## Citation
If you find our paper or code useful for your research, please cite:
```
@InProceedings{Jing_2021_ICCV,
    author    = {Jing, Junpeng and Deng, Xin and Xu, Mai and Wang, Jianyi and Guan, Zhenyu},
    title     = {HiNet: Deep Image Hiding by Invertible Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4733-4742}
}

```
