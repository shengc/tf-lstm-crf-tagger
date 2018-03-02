This is the TensorFlow implementation based on the paper, "[Neural Architecture for Named Entity Recognition](https://arxiv.org/abs/1603.01360)". It also heavily borrows the idea from its Pytorch implementation from [here](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf).

The purposes of creating this implementation are,

* to help me better understand the difference between PyTorch and TensorFlow, so I can learn from both framework most consciously.
* the neural network struture proposed by the paper has been proven a strong competitor for any tasks that has nature in sequence learning, including NER, POS, Semantic Role Labeling, etc. TensorFlow is a dropin choice for _amost_ any production environment. One can even deploy a TensorFlow model to a JVM process, which can be quite helpful for my future work.

There are gotcha and caveat for this implementation,

* the PyTorch implementation is kind of intuitive (depending on your familarity of CRF and Viterbi Coding). However to convert it to Tensorflow is non trivial. I had to use lots of higher order functions (map\_fn, foldl, scan) to avoid creating Tensors on the fly, given the fact sequence data is inherently undeterministic in its length. As a result, one can plug in any optimizer, such as AdamOptimizer, to train the model.
* I have not been able to implement the mini-batch version. I think it should be possible, as long as TensorFlow allows for nested higher order function. On the other hand, I think this should be much easier for PyTorch, though its implementation from the above link does not support mini batch either.

The original paper discusses the benefits of adding character level embedding and dropout to improve the performance. It is interesting to explore that possiblity later.
