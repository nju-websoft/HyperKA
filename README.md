# HyperKA
Source code for EMNLP-2020 paper "[Knowledge Association with Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/pdf/2010.02162.pdf)".

> Capturing associations for knowledge graphs (KGs) through entity alignment, entity type inference and other related tasks benefits NLP applications with comprehensive knowledge representations. Recent related methods built on Euclidean embeddings are challenged by the hierarchical structures and different scales of KGs. They also depend on high embedding dimensions to realize enough expressiveness. Differently, we explore with low-dimensional hyperbolic embeddings for knowledge association. We propose a hyperbolic relational graph neural network for KG embedding and capture knowledge associations with a hyperbolic transformation. Extensive experiments on entity alignment and type inference demonstrate the effectiveness and efficiency of our method.

## Datasets
We use three datasets in our experiments, i.e., [DBP15K](https://github.com/nju-websoft/JAPE) for entity alignment, [YAGO26K-906](https://github.com/JunhengH/joie-kdd19) and [DB111K-174](https://github.com/JunhengH/joie-kdd19) for type inference. We provide the datasets in the folder [./dataset/](https://github.com/nju-websoft/HyperKA/tree/main/dataset) of the repository.

## Code

### Package Description

```
src/
├── hyperka/
│   ├── ea_apps/: implementations for entity alignment
│   ├── ea_funcs/: implementations of training and test functions for entity alignment
│   ├── et_apps/: implementations for entity type inference
│   ├── et_funcs/: implementations for training and test functions for entity type inference
│   ├── hyperbolic/: implementations for hyperbolic operations
```

### Dependencies
* Python 3.6
* Tensorflow 1.14
* Scipy
* scikit-learn
* Numpy
* Pandas
* Matplotlib
* psutil

### Installation
We recommend creating a new conda environment to install and run HyperKA. You should first install Python 3.6 and Tensorflow-GPU 1.14 using conda. 
Then, HyperKA can be installed using pip with the following script:
```bash
pip install -e . -i https://pypi.python.org/simple
```

### Running
For example, to run HyperKA (75 dim) on ZH-EN of DBP15K for entity alignment, please use the following commands:
```bash
cd src/hyperka/ea_apps/
python main.py --input ../../../dataset/dbp15k/zh_en/mtranse/0_3/
```

For example, to run HyperKA (75/25 dim) on DB111K-174 for entity type inference, use the following commands:
```bash
cd src/hyperka/et_apps/
python main.py --input ../../../dataset/joie/db/ --neg_typing_margin 0.1 --neg_triple_margin 0.2 --nums_neg 30 --mapping_neg_nums 30 --batch_size 20000 --epochs 100
```

> Due to the instability of optimizing hyperbolic embeddings, it is acceptable that the results fluctuate a little (±1%) when running code repeatedly. You can run the code several times and choose the average result.

> If you have any difficulty or question in running code and reproducing experimental results, please email to zqsun.nju@gmail.com or cmwang.nju@gmail.com.

## Citation
If you use our model or code, please kindly cite it as follows:      
```
@inproceedings{HyperKA,
  author    = {Zequn Sun, 
               Muhao Chen,  
               Wei Hu, 
               Chengming Wang, 
               Jian Dai, 
               Wei Zhang}, 
  title     = {Knowledge Association with Hyperbolic Knowledge Graph Embeddings}, 
  booktitle = {EMNLP}, 
  year      = {2020}
}
```
