# Deepforest_Trials
Repository contains notebook tutorials that combine the deep forest machine learning model with the data obtained from the project. 

## DeepTree prediction experiment

[Results plots](https://nlesc.sharepoint.com/:f:/s/XAI4GEO/ElGyoojqYWRAqInJ1wlrijkBwGob-afGV5LXGkkb5Y9ULw?e=WwfzEd)

Experiment of predicting tree canopies with `predict_tile` function with different input arguments. Performed in the notebook:  **Deepforest_predict_exp.ipynb**. 

The variables are:

- **Crop size**: the size of the crop from the original UAV image. Predictions are performed within each crop. In the experiment 800x800 and 2000x2000 were tested.
- **Tile size**: Each crop are chopped to smaller tiles with overlaps. The predictions are done within the tile. In the experiment tile sizes from 300 to 1400 were tested.
- **Tile overlap size**: In percentage. Overlapping  sizes from 0.25 to 0.65 were tested.
- **use_soft_nms**: Binary switch of Gaussian Non-Maximum Suppression. When turned on, the confidence of *intersection-over-union threshold* will be reduced. Hence the algorithm tends to merge less boxes. See the documentation of DeepTree.

Conclusions testing different parameters with `predict_tile`:

1. Optimized tile sizes are between 500 and 800
2. Increasing the overlap ratio can in general improve the prediction results, but the relative size between the tile and the crop should allow this.
3. NMS does not improve the results.
4. The DeepTree model seem to be trained on the "round shaped" canopies. In general it misses some "palm tree" like trees.