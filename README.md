# Visual Question Answering with Module Networks

![alt text](https://github.com/omar-florez/VQA-ModuleNetworks/blob/master/output/00000076.jpg)

This work visualizes is based on [1](https://github.com/ronghanghu/n2nmn) and visualizes the Question Answering capabilities of Module Networks. Language guides the generation of neural architectures that maximizes the likelihood of answering a question correctly when correlated with visual embeddings.  

* A seq2seq architecture translates open questions into a sequence of available modules (Age, Gender, Emotion, Find, Transform, Locate, And, Describe, etc.) whose In-order traversal represents a hierarchical relation between modules. 25,050 unique questions generate hierarchical module networks. Some modules receive visual and language features while others receive attention maps.

* The *res5c* layer from [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) pretrained on ImageNET produces embeddings vectors of (1, 14, 14, 2048).


[1] R. Hu, J. Andreas, M. Rohrbach, T. Darrell, K. Saenko, *Learning to Reason: End-to-End Module Networks for Visual Question Answering*. in arXiv preprint arXiv:1704.05526, 2017. ([PDF](https://arxiv.org/pdf/1704.05526.pdf))
```
@article{hu2017learning,
  title={Learning to Reason: End-to-End Module Networks for Visual Question Answering},
  author={Hu, Ronghang and Andreas, Jacob and Rohrbach, Marcus and Darrell, Trevor and Saenko, Kate},
  journal={arXiv preprint arXiv:1704.05526},
  year={2017}
}
```

