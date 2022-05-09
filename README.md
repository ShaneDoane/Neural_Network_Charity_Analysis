# Neural_Network_Charity_Analysis
Using neural networks to determine which charities make good use of donations from our organization. This will let us optimize future donations.
Resources used: Jupyter Notebooks, Python, Pandas, Scikit-learn, TensorFlow, Google Colab Notebooks.

## Overview
The purpose of this analysis was to predict which charities were considered successful donations, for our donating organization: AlphabetSoup. Successful donations were ones that were used apporpriately by the receiving charity. There were labeled outcomes, shown under IS_SUCCESSFUL in this preview of the csv after initial load. Though we could perform supervised ML techniques like logistic regression to predict the probability of successful donations, neural networks are useful for this application, since there are many unique values for these categorical variables, such as APPLICATION_TYPE.

![image](https://user-images.githubusercontent.com/93338132/167333607-c7d4d517-bd0f-4e7a-a3ce-2dc8140add3b.png)

Unique values for each
![image](https://user-images.githubusercontent.com/93338132/167333875-45060ca4-3fc0-41b3-aa6e-f8db8d1adc0a.png)


## Results 

### Data Preprocessing
* The target variable is "IS_SUCCESSFUL"
* The features are all 44 scaled variable columns after 1) binning uncommon unique values into 'other' buckets, and 2) encoding categorical variables using OneHotEncoder (example below)

![image](https://user-images.githubusercontent.com/93338132/167334130-e2b4177a-95e8-4e20-8f69-f2adb5d791b6.png)

* Variables not used as feature or targets: 'EIN', 'NAME'. This is because they are identifiers, not relevant variables.

### Compiling and Training Models
Target performance is 75.0% accuracy.
All NNs ran 50 epochs, saving checkpoints every 5 epochs.

#### Initial Model
* Layers: 2
* Neurons by layer: 80, 30
* Activation functions (hidden layers): relu
* Activation functions (output layer): sigmoid
* Accuracy Achieved: 73.0%
* Achieved target?: No

2 layers were chosen with 80 and 30 neurons respectively because there were 44 input variables and 2x the inputs is a rule of thumb, but 80 is a nice round number, and felt appropriate if adding another layer. 30 was chosen to keep the paramaters lower than layer 1.

![image](https://user-images.githubusercontent.com/93338132/167335395-c841cd7c-36d1-467d-8674-8ee5bb49c69a.png)


![image](https://user-images.githubusercontent.com/93338132/167334571-a962add8-6740-4cd5-8d65-43bfd43f71b5.png)


#### Optimization Attempt 1
* Layers: 2
* Neurons by layer: 80, 30
* Activation functions (hidden layers): relu
* Activation functions (output layer): sigmoid
* Accuracy Achieved: 73.3%
* Achieved target?: No
* CHANGE - Dropped variables: 'Status', 'Special Considerations'

Due to the binary results and lack of variety in values for these two columns, I removed them from the dataset during preprocessing. This improved the model by 30 bps.

![image](https://user-images.githubusercontent.com/93338132/167334715-68fc2e55-735a-49fb-bd57-cb1ba961c23e.png)


#### Optimization Attempt 2
* Layers: 2
* CHANGE - Neurons by layer: 100, 40 from 80, 30
* Activation functions (hidden layers): relu
* Activation functions (output layer): sigmoid
* Accuracy Achieved: 73.0%
* Achieved target?: No
* From Opt 1 - Dropped variables: 'Status', 'Special Considerations'
* CHANGE - Increased size of 'Other' bin for Application_Type 

I added neurons to each layer to see if that improved on the improvements from Opt 1. The results suggested neurons do not improve accuracy with data in its current state. Though results are flat vs our original, the training time increased.

I also tightened the binning for Application_Type, requiring 1,000 counts instead of 500 to keep a given App Type as its own column. This was abandoned in subsequent models.

![image](https://user-images.githubusercontent.com/93338132/167335614-a9ff00a4-466d-475e-a91a-9d5c3a489739.png)


![image](https://user-images.githubusercontent.com/93338132/167335587-cc11f17c-28ed-45d5-8955-ad687d30a328.png)

#### Optimization Attempt 3
* CHANGE - Layers: 3
* REVERT and ADD - Neurons by layer: 80, 30, 10 
* Activation functions (hidden layers): relu
* Activation functions (output layer): sigmoid
* Accuracy Achieved: 73.0%
* Achieved target?: No
* From Opt 1 - Dropped variables: 'Status', 'Special Considerations'
* REVERT - Increased size of 'Other' bin for Application_Type 

There was no improvement from Opt 1 or Opt 2 with a third layer. I reverted the two changes from Opt 2 due to their lack of effects.

![image](https://user-images.githubusercontent.com/93338132/167335721-b0cf9c9d-590c-4107-984b-f13023308b08.png)

![image](https://user-images.githubusercontent.com/93338132/167335733-9c2539d3-4e7b-4ae5-98a1-f7ef4da8fbc1.png)


#### Optimization Attempt 4: "The Kitchen Sink"
* Layers: 3
* CHANGE - Neurons by layer: 100, 40, 15 
* CHANGE - Activation functions (hidden layers): sigmoid
* Activation functions (output layer): sigmoid
* Accuracy Achieved: 73.2%
* Achieved target?: No
* From Opt 1 - Dropped variables: 'Status', 'Special Considerations'

Here I added neurons to layers 1 and 2, per Opt 2. I kept the third layer from Opt 3 and added 5 neurons. I then changed activation functions in the hidden layers to sigmoid functions. This resulted in a 20 bps improvement from Initial Model.

<img width="627" alt="image" src="https://user-images.githubusercontent.com/93338132/167335986-24ab2acc-653f-4f2f-a994-f1e9d707b472.png">

<img width="652" alt="image" src="https://user-images.githubusercontent.com/93338132/167336007-1f345e2d-0e1a-48dc-babb-a20b767ffc3c.png">


## Summary
Overall, despite efforts to improve on the initial model to achieve 75% accuracy, these attempts were fruitless. Best attempt was 73.3%. We can translate this to 73% of donations being successful. While that may sound good, we still are wasting more than 1 of 4 donations! Ideally, we'd want to achieve 80% accuracy or more.

Another potential NN model to try would be many layers, each with fewer neurons. Another option would be a logistic regression, another binary classifier that is used for probability predictions.
