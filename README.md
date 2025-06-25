# Heidelberg IA & Health summer school

This repository contains the code for the 2nd workshop of the Heidelberg IA &
Health summer school.  

You can either use it locally by cloning the repo, installing the necessary
libraries with either `pip install .` or with `uv` and `uv sync`.  
Then you can look at the TD at `notebooks/TD.ipynb`.  

Or, you can directly use colab to do the TD at :

<a target="_blank" href="https://colab.research.google.com/github/etienneguevel/heidelberg/blob/main/notebooks/TD.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## White blood cells

**Give some context on the task**

### Object classification

Classification is a key task in hematology, as class distributions and the
total amount of white blood cells help the clinician to make a diagnostic.  

In this part we look at the composition of the dataset, how to import a
pretrained CNN model and fine-tune it on our data by training it.

### Foundation models

Foundation models are models that are not used directly to give the output we
want, but instead make *embeddings* out of the inputs we give them.  
Those embeddings are then used for downstream tasks such as classification.  

In this part we look at how to import a pretrained Vision Transformer, and how
to test the quality of its embeddings as well as use them for other tasks.
