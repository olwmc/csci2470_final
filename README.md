# FREEZE!: Pretrained Language Models as Frozen Feature Extractors for Semantic Tasks
## CSCI 2470 Final Project
See the final paper for a full description of the project.

### Data
Download activation data [here](https://drive.google.com/drive/folders/1LhINDDAmJZrvKMafVMSEjkbGgEQgFl93?usp=drive_link) or compute it yourself in the `generate_activations` notebook. Expect it to take several hours to generate.

### Notebook descriptions
- `Pretraining-data-gen.ipynb`  Generates and saves the activations for each of the 20,000 sentences extracted from GenericsKB and passed through qwen2
- `STS-LEAKAGE-IS-Meaningful.ipynb`  Exposes that within the STSB dataset the train and test sets have ~18% overlap
- `autoencoders.ipynb`  Shows the entire modeling process and forms the basis for further experimentation in `autoencoders_simple_alterations.ipynb` and `autoencoders_simple_alterations_with_Bayes.ipynb`
- `autoencoders-qwen.ipynb`  The same as `autoencoders.ipynb` but using qwen instead of gpt2-medium
- `autoencoders_simple_alterations.ipynb`  Shows the first manual hyperparameter experiments and investigation of intermediate correlations, also uncovers the bottleneck in the autoencoder
- `autoencoders_simple_alterations_with_Bayes.ipynb`  The same as `autoencoders_simple_alterations.ipynb` but with an automated hyperparameter optimization trial at the end
- `generate_activations.ipynb`  Generates and saves the activations for each sentence in the STS dataset passed through gpt2-medium
- `individual_sup_training.ipynb`  Demonstrates that training autoencoders individually on STS reaches nearly the same performance
- `investigating-intermediate-correlations.ipynb`  Exposes the varying autoencoder distance correlations over all layers and the magnitude shifts of standard deviation and mean over layers
- `nli-datagen.ipynb`  Generates and saves the activations for each sentence in the NLI dataset passed through gpt2-medium
- `nli-pretraining`  Shows that doing Sentence-Bert style pretraining has no effect on performance
- `projection_from_sup.ipynb`  Attempts to project encoder vectors into a higher dimensional space, add them together, and train that as the embedding vector
- `train_layerwise_siamese.ipynb`  Trains a layer-wise siamese network to test similarity learning significance
- `train_weighted_cosine.ipynb` Trains a model that boils down to weighting the pairwise cosine distance at each layer
