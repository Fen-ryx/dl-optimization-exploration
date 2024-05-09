# Optimization in Deep Learning
The goal of this project is to find something that works better than the best, Adam. To this end, I am going to set up several different experiments to test out current optimizers and figure out where they fail. That is what this folder is for.

## First Experiments
I work with various existing optimizers: Adam, AdaGrad, RMSProp, SGD, SGD with Momentum, etc. on Vanilla Neural Networks to see which performs best. The task is simple classification (MNIST).<br>
I am going to record the following parameters:<br>
<ol>
    <li> Number of epochs and steps taken till convergence </li>
    <li> Time taken till convergence </li>
    <li> Performance on the final task (accuracy) </li>
    <li> Final training loss when training is terminated (early stopping used with a patience of 3 epochs) </li>
</ol>