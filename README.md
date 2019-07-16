# Decision-Tree-vs-Boosting
Exploration of why gradient boosting performs better than decision trees

The code included here compares the step functions that a decision tree regressor and a gradient boosting regressor fit to a specified function. The decision tree appears to try to approximate the target function pointwise (with sufficient depth), while the gradient boosting regressor seems to overfit less because it tries to approximate in the mean square sense.

I will soon add a document explaining these observations in detail.
