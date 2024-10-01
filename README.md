# Code for "Plug-and-Play Gibbs sampler: Embedding Deep Generative Priors in Bayesian Inference"

### 1) Install packages

````
conda create -n pnpsgs python=3.10
conda activate pnpsgs
pip install -r requirements.txt
````

### 2) Downlad pretrained diffusion models

Download the corresponding checkpoint from the links below and move it to ./models/.

- [FFHQ](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)
- [Imagenet](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)

### 3) Run experiments

For now, you can just launch the notebooks corresponding to the targeted task. 

To DO :
- Inpainting 
- Super-Resolution
- Neat configs files to run experiments efficiently with metrics

### 5) Citation 

Thank you for your interest in our work! Please consider citing

````
@article{coeurdoux2024pnpsgs,
  author={Coeurdoux, Florentin and Dobigeon, Nicolas and Chainais, Pierre},
  journal={IEEE Transactions on Image Processing}, 
  title={Plug-and-Play Split Gibbs Sampler: Embedding Deep Generative Priors in Bayesian Inference}, 
  year={2024},
  volume={33},
  pages={3496-3507},
  doi={10.1109/TIP.2024.3404338}
}
````
