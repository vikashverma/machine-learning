### Decision Tree

#### Entropy(Randomness, How unpure data is)

- -sum (p * log2(p))

#### Information Gain(0 to 1)

- entropy(parent) - [weighted average] entropy(children)

> Decision tree algorithm tries to maximize information gain.
>
> This is how DT will choose a feature to split on



> Ensemble method is based on Decision tree



#### Neural Networks

> If data is linearly separable then perceptron rule will definitely find it. It just needs certain number of iteration. 



#### SVM

> The basic idea of finding the line of least commitment in the linear separable set of data is the basis behind support vector machine.
>
> SVM finds a  line, or in more general a hyperplane between data of two classes.

- First correctly classify the labels
- Then Maximize the margin



> y = w^t x + b

>Machine that only needs a few support vectors

> **Kernel Trick**: What really happens is projection of data points in some higher dimensional space where data is linearly separable. But since we use kernel function to represent domain knowledge, we don't actually need to do computation of transforming the data into higher dimensional space.

e.g. K = (X^T Y)^2, K = X^T Y, K = (X^T Y + c)^p



**Mercer Condition**: It acts like a distance or it acts like a similarity



#### KNN

- Lazy learner whereas linear regression is eager learner
- **J**ust **i**n **T**ime **L**earning(JITL or JIL) 
- Has  Preference Bias


**The curse of Dimensionality**: As the amount of feature grows or dimension rows, the amount of data we need to generalize accurately grows exponentially



#### Naive Bayes

> **Naive**, because it does not consider the order of the words

- **MAP**: Maximum a Posteriori
- **ML**: Maximum Likelihood Hypothesis





##### Joint Distribution

##### Belief Networks



##### Ensemble Learning: Boosting

- Learn over a subset of data -> Rule

- Combine -> Complex Rule

  ---

  ​

- **Bagging**: Random subset combined by mean, also called **Bootstrap Aggregation**



- Pink Noise: Uniform Noise
- White Noise: Gaussian Noise
