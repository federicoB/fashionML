Inspired by Zalando [Fashion DNA](https://research.zalando.com/project/fashion_dna/fashion_dna/) encoding, I wanted to try to recreate it by also using the opportunity offered by their released [Feidegger](https://github.com/zalandoresearch/feidegger) dataset.

I implemented the code to optimally download the dataset, and I firstly implemented a Linear Autoencoder (with disastrous results). The outcome was way better with a Convolutional autoencoder. I then used the learned encoding and a KDtree to search similarities between articles, and the queries results are shown in the following pictures.

I then moved to develop a DCGAN architecture both for generation and encoding, but without success for now, as highlighted in this [issue](https://github.com/federicoB/fashionML/issues/3).

![image](https://user-images.githubusercontent.com/15829877/145912334-68cf283c-3a51-4f94-9960-234c7e5aa046.png)

![image](https://user-images.githubusercontent.com/15829877/145912377-2046a3ee-f491-4873-a066-fdcf07282f21.png)

![image](https://user-images.githubusercontent.com/15829877/145912396-9d2fc8d6-da70-4a46-bf15-4f1a60baac70.png)
