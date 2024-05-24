# GraphSmin
This is the code for GraphSmin.
## Overview
Dissolved Gas Analysis (DGA) is a widely adopted technique for detecting faults in oil-immersed power transformers. However, in the context of DGA fault diagnosis, the fault class
has considerably fewer samples compared to the normal class. Directly constructing fault
diagnosis model in such an imbalanced scenario may result in insufficient representation of
fault classes, which can ultimately lead to a decrease in diagnostic performance. To tackle
this issue, we propose a novel imbalanced DGA model called GraphSmin. This approach is
equipped with contrastive dual-channel graph filters and deliberately generate minority samples
to effectively address the imbalance problem in fault diagnosis. Specifically, similar KNN graph
(S-KNN) and dissimilar KNN graph (DS-KNN) are established to better reveal the complex
relationship between samples. Subsequently, the dual-channel graph filters with contrastive
learning is presented to obtain high-quality embeddings of DGA samples. In particular, we
expand minority class samples in the embedding space to ensure effective learning of minority
class features. Besides, an edge predictor is trained to model the relationship information
between nodes. Extensive experiments on two datasets demonstrate the outstanding capability
and reliability of the proposed method in imbalanced DGA fault diagnosis. 
<div  align="center">    
    <img src="./assets/framework.png" width=90%/>
</div>
<div  align="center">    
      Figure 1: Illustration of the proposed GraphSmin.
</div>