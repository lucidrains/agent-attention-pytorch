<img src="./agent-attention.png" width="350px"></img>

## Agent Attention - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2312.08874">Agent Attention</a> in Pytorch.

This work seems to be an elegant simplification of <a href="https://github.com/lucidrains/isab-pytorch">`ISAB`</a> architecture from the <a href="https://arxiv.org/abs/1810.00825">Set Transformers</a> paper (requires only one attention block rather than two). While ISAB works, I have found it to be a bit unstable, thus wondering if the simplification in this work resolves that issue.

This repository will add support for variable sequence lengths (masking) and post-softmax talking heads.

## Citations

```bibtex
@inproceedings{Han2023AgentAO,
	title 	= {Agent Attention: On the Integration of Softmax and Linear Attention},
	author 	= {Dongchen Han and Tianzhu Ye and Yizeng Han and Zhuofan Xia and Shiji Song and Gao Huang},
	year 	= {2023},
	url 	= {https://api.semanticscholar.org/CorpusID:266210414}
}
```

```bibtex
@misc{shazeer2020talkingheads,
    title   = {Talking-Heads Attention}, 
    author  = {Noam Shazeer and Zhenzhong Lan and Youlong Cheng and Nan Ding and Le Hou},
    year    = {2020},
    eprint  = {2003.02436},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
