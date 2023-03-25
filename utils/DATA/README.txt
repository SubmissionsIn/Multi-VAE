Data & Model Link：https://pan.baidu.com/s/1CHQ5EDgNPkW37uanDztGjQ 
Pass code：todz


Disentangling multi-view information&representation has a strict condition that it is clear what view-common information and view-peculiar/private information to be disentangled. In order to verify the disentanglement effectiveness of visual representations in Multi-VAE, we constructed two kinds of datasets as follows:

1. Multiple views are with view-common visual information and with different view-peculiar visual information. Such samples are constructed by the same datasets, where the view-common information among multiple views is the same sample object; and the view-peculiar information among multiple views is their different visual patterns. For convenience, these datasets are entitled Multi-MNIST, Multi-Fashion, Multi-COIL-10, and Multi-COIL-20.

2. Multiple views are with view-common semantic information but with different view-peculiar visual information. Such samples are constructed by different datasets, where the view-common information among multiple views is the same semantic category/label; and the view-peculiar information among multiple views is that they have different visual patterns coming from unrelated datasets. For convenience, these datasets are entitled Digit-Product and Object-Digit-Product.

It should be noted that:

1. Traditional multi-view datasets may not meet the requirement that multiple views contain view-common information and view-peculiar information that is obvious and can be disentangled. They may not be suitable for the model Multi-VAE as it focuses on disentanglement.

2. If one needs to use the datasets we constructed, please refer to the original source of these datasets, as described in many previous publications.
