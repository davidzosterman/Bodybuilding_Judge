# Bodybuilding_Judge
--------------------

Description here

Nice header image:
<img src="images/placeholder.png" alt="Nice Header Plot" width="300"/>

## Table of Contents
--------------------
- [Data Description](#data-description)
    - [Data scraping](#data-scraping)
    - [Data cleaning](#data-cleaning)
    - [Data processing](#data-processing)
- [Model](#model)
    - [Original two-input convolutional neural network](#Original-two-input-convolutional-neural-network)
    - [ResNet50](#ResNet50)
- [Results](#results)
    - [Name_this_section](#name_this_section)

## Data Description
-------------------
### Data scraping
Data consisted of photos from the [National Physique Committee website](https://contests.npcnewsonline.com/contests/2024). Photos were downloaded with the [data_scraping.ipynb](./notebooks/data_scraping.ipynb) Google Colab notebook.

### Data cleaning
To reduce training time and needed model complexity, we focused on competitions from the `year` 2024 in the `division` 'Men’s Physique'. Within each competition, each contestant has a variable number of photos. We manually selected two photos, one front and one back, from each contestant using the [data_selection.ipynb](./notebooks/data_selection.ipynb) Jupyter notebook (note that Google Colab did not support the interactive feature of this notebook). Each competition also listed the final `rank` of the competitors from 1 up to 16. If more than 16 competitors were present, all after the first fifteen were tied at 16th place.

The training, validation, and test datasets were created using the ???WHATE NOTEBOOK??? notebook. The purpose of our model was to compare an image from each of two competitors and decide a winner, so our dataset input `x` consisted of pairs of two photos from different competitors and our dataset `y` output consisted of an integer 0 (first competitor wins) or 1 (second competitor wins). Within each competition, we added a datapoint for each pair of competitors, using the rankings within the competition to determine if the winner should be 0 or 1. Two important nuances were:
- Each pair of competitors added *two* datapoints for each competitor pair, with the order of the competitors as competitor 0 or 1 reversed. This was to prevent the network's two image recognition parts from learning features of different competitors.
- If two competitors had the same ranking (eg. two competitors with ranking 16 or one competitor paired against himself) then the datapoint was thrown out.

We divided the data into [training](./data/df_train.pkl), [validation](./data/df_val.pkl), and [test](./data/df_test.pkl) sets, with a split of ???SPLIT OF WHAT???. The training set consisted of 11,776 image pairs with one-hot-coded labels. To keep the integrity of our validation and test sets, we kept competitions intact and ensured no competitor appeared in more than one of the three sets.

### Data processing
The photos, originally 600x900, were transformed to 128x128 using `torchvision.transforms.Resize()`.

## Model
-----------------------------------------
We employed two different models, an original two-input convolutional neural network and ResNet50.

### Original two-input convolutional neural network
We tried a hand-written model called TwoInput Net, which consists of three Conv2d() layers with ReLu() and MaxPool2d() functions between them. For every competition in the training set, we gave the model all pairs of pictures that had different rankings and a one-hot-coded variable indicating which picture was the winner. We optimized using the BCELoss() function.

### ResNet50
Text

## Results
-----------------------------------------------------------------------------------------------------------------------
Add text

### Name this section
Add text
