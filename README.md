# **Linear-Classification**
**Score function** maps raw data to class scores
**Loss function**  quantifies the agreement between the predicted scores and the ground truth labels

## Parameterized mapping from images to label scores
  let $x_i$ be image of ith index which is vector
  
  $x_i \in \mathbb{R}^D $\
  $i = 1, \dots, N$ \
  $y_i \in 1, \dots, K$\
\
  $N$ = no. of of pixels in a image\
  $D$ = no. of classes 

## Linear classifier

  **Score function** $f:\mathbb{R}^D -> \mathbb{R}^K$, maps the raw image pixels to class scores.

  $f(x_i, W, b) = Wx_i + b$

  The matrix $W$ (of size [K x D]), and the vector $b$ (of size [K x 1]) are the parameters of the function\
  The parameters in W are often called the weights, and b is called the bias vector

  >![image](https://github.com/user-attachments/assets/3a29d87f-c636-4f35-9615-42b630e0a3a1)
>An example of mapping an image to class scores. For the sake of visualization, we assume the image only has 4 pixels (4 monochrome pixels, we are not considering color channels in this example for brevity), and that we have 3 classes (red (cat), green (dog), blue (ship) class). (Clarification: in particular, the colors here simply indicate 3 classes and are not related to the RGB channels.) We stretch the image pixels into a column and perform matrix multiplication to get the scores for each class. Note that this particular set of weights W is not good at all: the weights assign our cat image a very low cat score. In particular, this set of weights seems convinced that it's looking at a dog.
>
Another interpretation for the weights W
 is that each row of W
 corresponds to a template (or sometimes also called a prototype) for one of the classes
 >![image](https://github.com/user-attachments/assets/998999ed-3b74-47b8-ad47-366e3eb8b67a)

### Bias trick 
  We can write 
  $f(x_i, W, b) = Wx_i + b$\
  as\
  $f(x_i, W) = Wx_i$

  by extending the vector $x_i$
 with one additional dimension that always holds the constant 1, a default bias dimension. With the extra column in $W$, the new score function will simplify to a single matrix multiply

>![image](https://github.com/user-attachments/assets/25bc753b-793a-46c8-9391-5bd8d7257b76)
>Illustration of the bias trick. Doing a matrix multiplication and then adding a bias vector (left) is equivalent to adding a bias dimension with a constant of 1 to all input vectors and extending the weight matrix by 1 column - a bias column (right). Thus, if we preprocess our data by appending ones to all vectors we only have to learn a single matrix of weights instead of two matrices that hold the weights and the biases.

##Multiclass Support Vector Machine loss
The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin $\Delta$

 For example, the score for the j-th class is the j-th element: $s_j=f(x_i,W)_j$
. The Multiclass SVM loss for the i-th example is then formalized as follows:

$L_i=\sum_{j≠y_i}max(0,s_j−s_{y_i}+\Delta)$\
this can be written as\
$L_i=\sum_{j≠y_i}max(0,{W_{j}}^{T} x_i− {W_{y_i}}^{T} x_i+\Delta)$\

-threshold at zero max(0,−)
 function is often called the hinge loss.

 >
>![image](https://github.com/user-attachments/assets/9e13302c-ce93-4119-be62-003951b0ea1a)
>The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.

**Regularization** we wish to encode some preference for a certain set of weights W over others to remove ambiguity, as there are many W's possible.We can do so by extending the loss function with a regularization penalty R(W).\
The most common regularization penalty is the squared L2 norm that discourages large weights through an elementwise quadratic penalty over all parameters:\
$R(W) = \sum_i\sum_j W_{i,j}$\

the full Multiclass SVM loss becomes:\
$L = \frac{1}{N}\sum L_i + R(W)$


## Softmax classifier
In the Softmax classifier, the function mapping $f(x_i;W)=Wx_i$ stays unchanged, but we now interpret these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss that has the form:\

$L_i = \log \left( \frac{e^{f_{y_i}}}{\sum_j e^{f_i}} \right) $
###Softmax vs SVM

>![image](https://github.com/user-attachments/assets/d08919b8-567c-4dc1-a988-184caf34410a)
>

  
