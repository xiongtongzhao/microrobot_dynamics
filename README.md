## Numerical solver for "Chemotactic navigation in robotic swimmers via reset-free hierarchical reinforcement learning"
[![arXiv](https://img.shields.io/badge/arXiv-2408.07346-df2a2a.svg)](https://arxiv.org/pdf/2408.07346)
[![Python](https://img.shields.io/badge/python-3.7.17-blue)](https://www.python.org)


Tongzhao Xiong, Zhaorong Liu, Yufei Wang, Chong Jin Ong, Lailai Zhu 
Department of Mechanical Engineering, National University of Singapore
<hr style="border: 2px solid gray;"></hr>

This repository contains the numerical implementation we have developed to solve the low-Reynolds-number swimming hydrodynamics of a linear microswimmer and a ring counterpart in unbounded and bounded space. This implementation forms the emulated environment for reinforcement learning. 
### Environment
```
pip install virtualenv
virtualenv -p python3.7.17 myenv
workon myenv
pip install -r requirements.txt
```

### Supported cases
1. The flagellar swimmer with $10$ links and ameboid swimmer with $20$ links swim in unbounded space with and without a background flow. 'discretization.py' needs to be run first for the discretization of the swimmer. 
```
python discretization.py
```
Then 'calculate_v.py' can be imported as a module to obtain the locomotion of the swimmer and pressure on it.
```
import calculate_v
```

2. The flagellar swimmer with $10$ links and ameboid swimmer with $20$ links swim in a bounded quiescent environment. Here, 'discretization.py' and 'constriction_discrete.py' must be run first to discretize the swimmer and the boundary, respectively. 
```
python discretization.py
python constriction_discrete.py
```
Then 'calculate_v.py' can be imported as a module to obtain the locomotion of the swimmer.
```
import calculate_v
```

### Citation
PLease consider citing our [paper](https://arxiv.org/pdf/2408.07346) if you find it useful:
```bibtex
@article{xiong2024enabling,
  title={Enabling microrobotic chemotaxis via reset-free hierarchical reinforcement learning},
  author={Xiong, Tongzhao and Liu, Zhaorong and Ong, Chong Jin and Zhu, Lailai},
  journal={arXiv preprint arXiv:2408.07346},
  year={2024}
}
```
