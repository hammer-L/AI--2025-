### Forward

$x$: input

layer 1:	 $z^{(1)} = w^{(1)}x + b^{1}, \;a^{(1)} = ReLU(z^{(1)})$

layer 2:	 $z^{(2)} = w^{(2)}a^{(1)} + b^{2}, \;a^{(2)} = ReLU(z^{(2)})$

layer 3:	 $z^{(3)} = w^{(3)}a^{(2)} + b^{3}$

sofmax:	$p = sofmax(z^{(3)})$

Loss function:	$L = -\frac{1}{n} \sum^n_{i=1}y_ilog(p_i)$   # n分类



### Backward

​	1. softmax + CrossEntropy:	$\frac{\partial L}{\partial z^{(3)}} = p - y$

​	2.对layer 3 中的 $w^{(3)}$ 求偏导:	$\frac{\partial L}{\partial w^{(3)}} = \frac{\partial L}{\partial z^{(3)}} (a^{(2)})^T = (p-y)(a^{(2)})^T$

​		对layer 3 中的 $b^{(3)}$ 求偏导: 	$\frac{\partial L}{\partial b^{(3)}} = p-y$

​	3.对layer 2 中的 $a^{(2)}$ 求偏导:	$\frac{\partial L}{\partial a^{(2)}} = (p-y)(w^{(3)})$

​		对layer 2 中的 $z^{(2)}$ 求偏导:	已知 $ReLU'(z) =	\begin{cases} 1, \;z>0 \\ 0,\;z\le0 \end{cases} $

​														 则 $\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \circ ReLU'(z^{(2)}) $

​		对layer 2 中的 $ w^{(2)}$ 求偏导:	$\frac{\partial L}{\partial w^{(2)}} = \frac{\partial L}{\partial z^{(2)}} (a^{(1)})^T $ 

​	4.对layer1同理

​	5.计算完偏导后，更新参数：	$w^{(i)} = w^{(i)} - \alpha \frac{\partial L}{\part w{(i)}}$



​			



