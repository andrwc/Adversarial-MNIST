# Adversarial examples: MNIST vs a deep CNN
Our purpose here is pretty straightforward: we'll fit a deep CNN to some MNIST data, generate adversarial examples from it for a specific digit, and analyze the results.

## Relevant Background
For the unfamiliar, [Breaking Convets](https://karpathy.github.io/2015/03/30/breaking-convnets/) is a great place to get started. 

From the literature, we'll lean on [Goodfellow et al](https://arxiv.org/abs/1412.6572) to get things done here, but you'll want to check out [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) as well. 

## Tooling and Env
We'll use AWS to speed up all of the compute intensive tasks. It's cheap, fast, and why not. What a time to be alive. Specifically:

```
Region: us-east-1b
AMI: ubuntu14.04-cuda7.5-tensorflow0.9 - ami-81960a96 
```

Instance Type|ECUs|vCPUs|Memory (GiB)|Instance Storage (GB)|EBS-Optimized Available|Network Performance
---|---|---|---|---|---|---
g2.2xlarge|26|8|15|1 x 60|Yes|High

We're using the python bindings to tensorflow provided by the AMI (which has cuda enabled). I [pip froze](./pip.freeze) my stuff too if you're super curious. We'll do other stuff locally but the env's are essentially the same. 

The boilerplate for the deep CNN can be found [here](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html), borrowing as we go from [*CLEVER HANS*](https://github.com/openai/cleverhans) :+1: and [Keras](https://keras.io/) -- but the real heavy lifting comes from the reference implementation for adversarial example generation in Clever Hans (written by Goodfellow himself, no less).

## Results
Our CNN performance came out exactly as expected per the tutorial: ~0.992-0.993 accuracy on the test data for non-adversarial examples. 

We'll generate our adversarial examples only from the '2' class of handwritten digits, extracted from the train, validation, and test sets (6990 2's in total in MNIST). Then try to mangle them in such a way as to cause them to be misclassifed as 6's. The process of extracting these examples doesn't change our model params in any way, so we don't need to worry about how our data is partitioned -- train, test, and validation are all fair game.

We have a single tunable hyperparameter epsilon (per the Goodfellow paper), which we can consider to be a lever controlling how *insidious* an adversarial example is. The more insidious, the less perceptible the changes are to the human eye that cause the example to be misclassified. In that spirit, a conservative value for epsilon was taken, but as mentioned in the paper misclassifications are stable across a wide range of epsilon for a given adversarial example, so optimizing for eps here felt a little... premature, regardless.

### The takeaway
**2's don't want to be 6's very much**. Only 54 of our 6990 adversarial 2's were classified as 6's with high confidence (that is, confidence exceeding all of the other confidence scores for the other classes. ARGMAX style confidence). [**Here** in the comparison.html](https://rawgit.com/andrwc/Adversarial-MNIST/master/comparison.html) are the originals, alongside their adversarial partners. Apart from the subtle artefact-y distortion these are quite insidious to my eye. 

So what about the other 6936 adversarial 2's that didn't fall into the 6's bin? The appendix of the Goodfellow paper gives us some idea. It's not totally suprising that highly performant classifiers are harder to trick adversarially, but it also turns out 2's don't mangle easily into 6's. Here is the distribution of the predicted classes for our adversarial examples:

![class frequencies](./class_frequencies.png?raw=true)

As you can see most of our adversarial 2's, remained 2's. 8's are also popular, sharing quite a bit of pixel real-estate on average with 2's. 6's were among the least common.

The distribution of the scores per class is as follows:

![box plot](./box.png?raw=true)

Bear in mind here that we're looking at the distribution over all scores, ignoring the argmax. Taking another view of the support:

![histogram](./hist.png?raw=true)

we can see that the model scores our adversarial examples in a pretty even handed way. Everything nice, unimodal, symmetric. Not what you might expect considering that these examples were engineered specifically to *exploit weaknesses* in the classifier's structure related to it's training. But nope, nothing wild here (and perhaps thanks to our choice of epsilon), we see a sharp drop in overall accuracy (from 99.3% to ~58%) but the ones it misclassifies it doesn't overheat on.. just treats them as vaguely familiar and uncertain.

## Files
The deep CNN code for training can be found in `deep_mnist_example_code.py`, otherwise check `computational_graph.py`. Our primary work is found in `adversarial_mnist.py` for filtering and generating images.

`utils.py` contains the relevant source from Clever Hans and Keras, modified for our purposes. Lastly `analysis.py` does the analysis and viz.

The included `*.npy` files, which admittedly make the repo a lil big, contain variously, the mnist 2's, adversarial 2's and the model predictions for each adversarial 2. You'll need them to run `analysis.py` locally, which doesn't depend on tensorflow.

## The End
