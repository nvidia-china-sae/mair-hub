# Recipe of LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Model

This repository includes the recipe of how to finetune the Speech Foundational Model (SFM) with **L**arge Language Model **E**nhanced **S**emi-**S**upervised Learning (LESS) method.  

## General overview

Pipeline is illustrated in the below figure, taking the Spanish-to-English Automatic Speech Translation (ES-to-EN AST) as an example.  We have several steps: 

 1. Finetune the SFM at T=0 with supervised  data
 2. Prepare the unsupervised data 
 3. Inference the unsupervised using the initial SFM (SFM 0) to get the pseudo labels
 4. Refine the pseudo labels with an LLM, and perform data filtering using proper **hypo_wer**
 5. Combine the supervised data and pseudo-labeled unsupervised data together, and finetune the SFM. Go step 3 and iterate until converge
 
<p align="center">
<iframe src="assets/less_pipeline.pdf" height="500" frameborder="0" />
</p>

## Steps
### Prepare the unsupervised data
### Generate the pseudo labels
### Perform LLM requests
### Data filtering
### Finetune the SFM with both supervised and unsupervised data

## Results
## Citation
``` bibtex
@misc{ding2025lesslargelanguagemodel,
      title={LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models}, 
      author={Wen Ding and Fan Qian},
      year={2025},
      eprint={2506.04586},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04586}, 
}
