# DSLR-release
This repository is a Pytorch implementation of the paper [**"DSLR: Deep Stacked Laplacian Restorer for Low-light Image Enhancement"**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9264763)

Seokjae Lim and [Wonjun Kim](https://sites.google.com/view/dcvl)  
IEEE Transactions on Multimedia

When using this code in your research, please cite the following paper:  

Seokjae Lim and Wonjun Kim, **"DSLR: Deep Stacked Laplacian Restorer for Low-light Image Enhancement,"** **IEEE Transactions on Multimedia** **DOI:**[**10.1109/TMM.2020.3039361**](https://ieeexplore.ieee.org/document/9264763).


### Experimental results with state-of-the art methods on the MIT-Adobe FiveK dataset
![example1](./example/fig1.png)
Test samples from the MIT-Adobe FiveK dataset and corresponding enhancement results by previous methods and the proposed DSLR. (a) Origianl input. (b) CLAHE. (c) LDR. (d) LIME. (e) HDRNet. (f) DR. (g) DPE. (h) UPE. (i) DSLR (proposed). (j) Ground truth

### Experimental results with state-of-the art methods on the MIT-Adobe FiveK dataset
![example2](./example/fig2.png)
More examples of enhancement results on the MIT-Adobe FiveK dataset. (a) Origianl input. (b) CLAHE. (c) LDR. (d) LIME. (e) HDRNet. (f) DR. (g) DPE. (h) UPE. (i) DSLR (proposed). (j) Ground truth

### Experimental results with state-of-the art methods on our own dataset
![example3](./example/fig3.png)
Test samples from our own dataset and corresponding enhancement results by previous methods and the proposed DSLR. (a) Origianl input. (b) CLAHE. (c) LDR. (d) LIME. (e) HDRNet. (f) DR. (g) DPE. (h) UPE. (i) DSLR (proposed).

### Experimental results with state-of-the art methods on our own dataset
![example4](./example/fig4.png)
More examples of enhancement results on our own dataset. (a) Origianl input. (b) CLAHE. (c) LDR. (d) LIME. (e) HDRNet. (f) DR. (g) DPE. (h) UPE. (i) DSLR (proposed).


### Requirements

* Python >= 3.5
* Pytorch 0.4.0
* Ubuntu 16.04
* CUDA 8 (if CUDA available)
* cuDNN (if CUDA available)

### Pretrained models
You can download pretrained DSLR model
* [Trained with MIT-Adobe FiveK dataset](https://drive.google.com/file/d/1bBUHzbjG6E--8o5SzCerJIUp9Q1RnXIY/view?usp=sharing)

### Note 
1. you should place the weights in the ./data/model/ 
2. Dataset is also placed in the ./data directory  (i.e., ./data/training_dataset)
3. Testset is placed in the ./data/test/input directory
4. test results are saved in the ./data/result/

### Training
* DSLR: Deep Stacked Laplacian Restorer training
```bash
python main.py n
```
## Testing 
* DSLR: Deep Stacked Laplacian Restorer testing
```bash
python test.py t
```
