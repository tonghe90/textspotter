# An End-to-End TextSpotter with Explicit Alignment and Attention

This is initially described in our [CVPR 2018 paper](https://arxiv.org/abs/1803.03474).

<img src='imgs/horse2zebra.gif' align="right" width=384>


## Getting Started
### Installation
- Clone the code
```bash
git clone https://github.com/tonghe90/textspotter
cd textspotter
```

- Install caffe. You can follow this [this tutorial](http://caffe.berkeleyvision.org/installation.html)
```bash
# make sure you set WITH_PYTHON_LAYER := 1
cp Makefile.config.example Makefile.config
make -j8
make pycaffe
```

- install editdistance: `pip install editdistance`

- After Caffe is set up, you need to download a trained model (about 40M) from [Google Drive](https://arxiv.org/abs/1803.03474)
- Run `python test.py --img=./imgs/img_105.jpg`



## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{tong2018,
  title={An End-to-End TextSpotter with Explicit Alignment and Attention},
  author={T. He and Z. Tian and W. Huang and C. Shen and Y. Qiao and C. Sun},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018}
}

```