print("The dimension of the embeddings of the model is:", model.embed_dim)
print(f"\nThere are {sum([p.numel() for p in model.parameters()]):.3g} parameters in our model.")