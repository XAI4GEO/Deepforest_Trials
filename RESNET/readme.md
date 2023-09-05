# Transfer Learning with ResNET
Repository contains deep learning model trial using transfer learning for individual tree crown (ITC) classification. The training data from the [IDTreeS competition](idrees.org/competition/) is used to train the chosen pre-trained model and the model is tested with a validation data split from the training data-set. 

Experiments_metadata.xlsx - this file contains the details of all the experiments conducted using different model/data parameters as well as the jupyter notebook filename for the corresponding experiment.

General Conclusions:
- ResNET18 provided better results over ResNET50 which tended to overfit to the data
- Due to the limited data size, adding additional layers to the model led to overfitting
- Data augmentation helps improve the results overall
- SMOTE does not necessarily improve the results of the model. 
- Best experiment: experiment11.ipynb with resnet18.  