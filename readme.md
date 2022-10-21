# Personalized PCA

This is the implementation of the paper *Personalized PCA: decoupling shared and unique features*.

If you run into any problems, please submit an issue, or contact the authors of that paper.

## How to test a simple example
- run python3 ppca.py --dataset=borrowpowertest --logoutput=True

Results will be generated to the "outputs/" folder

## How to run the video segmentation experiment
- Create a folder called "images/"
- Put all the image frames of the video in the "images/" folder
- Rewrite the first line of the function load_car_data in the file imgpro.py accordingly
- Run python3 ppca.py --dataset='img_test' --logoutput=True
- If needed, fine-tune 'ngc' and 'nlc' to get the best results

## How to run the presidential debate topics modeling experiment
- Download the dataset from [this link](https://www.kaggle.com/datasets/arenagrenade/us-presidential-debate-transcripts-19602020)
- Put all files in the "debate/" folder
- Run python3 ppca.py --dataset='debate_test' --logoutput=True

## File organization
- ppca.py contains all the hyperparameters
- algs.py contains the implementation of personalized PCA learning algorithm
- imgpro.py, vectorize.py, mnist.py are used to handle video data, debate corpus, and FEMNIST dataset respectively.
