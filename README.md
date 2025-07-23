<h1 align="center">Neural Surveillance: Unveiling the Enigma of Latent Dynamics Evolution through Live-Update Visualization</h1>
A live-update visualization method for monitoring the training process of DNNs!
<!-- Official source code for ESEC/FSE 2023 Paper: -->
<!-- <strong>DeepDebugger: An Interactive Time-Travelling Debugging Approach for Deep Classifiers</strong> -->

<p align="left">
    <a href='http://arxiv.org/abs/2405.15135'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
  </p>
</p>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Citation](#citation)

## Installation
### Dependencies
Please run the following commands in the command line:
```console
$ conda create -n sentrycam python=3.8
$ git clone https://github.com/xianglinyang/SentryCam.git
$ pip -r install requirements.txt
```
To install torch, please check [link](https://pytorch.org/get-started/locally/).

## Usage
Please see our [wiki]([https://github.com/xianglinyang/DeepDebugger/wiki](https://github.com/xianglinyang/DeepDebugger/wiki/How-to-use-DeepDebugger)) for more details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions, please feel free to reach out to me at xianglin@u.nus.edu.

## Reproducibility
```console
./simulation.sh
```

## Citation
If you find our tool helpful, please cite the following paper:
```bibtex
@inproceedings{yang2023deepdebugger,
  title={DeepDebugger: An Interactive Time-Travelling Debugging Approach for Deep Classifiers},
  author={Yang, Xianglin and Lin, Yun and Zhang, Yifan and Huang, Linpeng and Dong, Jin Song and Mei, Hong},
  booktitle={Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={973--985},
  year={2023}
},
@inproceedings{yang2022temporality,
  title={Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and Dong, Jin Song},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  year={2022}
},
@inproceedings{yang2022deepvisualinsight,
  title={DeepVisualInsight: Time-Travelling Visualization for Spatio-Temporal Causality of Deep Classification Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and He, Zhenfeng and Wang, Chao and Dong, Jin Song and Mei, Hong},
  booktitle = {The Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```







