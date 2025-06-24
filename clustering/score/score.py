import numpy as np
from tqdm import tqdm
import torch
import gc
# Load the whitened features
whitened = np.load("whitened.npy")

N = 100 * 8192
B = 16
d = 128*128

whitened = torch.from_numpy(whitened).to(torch.float32, memory_format=torch.contiguous_format).flatten()

for i in tqdm(range(8)):
	mm = np.memmap(f"../datacomp_feats/grads.fp16_part_{i}", dtype=np.float16, mode="r", shape=(N, B, d))

	print("loading")
	grads = torch.from_numpy(mm).to(torch.float32, memory_format=torch.contiguous_format).flatten(1)
	print("flattened + contiguous")
 
	# grads shape = (N, B*d)
 
	# Compute inner product of each row with whitened
	inner_products = torch.matmul(grads, whitened.unsqueeze(1)).squeeze(1)  # Shape: (N,)
	print("inner products")
	
	# Compute norms for cosine similarity
	grads_norms = torch.norm(grads, dim=1)  # Shape: (N,)
	whitened_norm = torch.norm(whitened)    # Scalar
	
	# Compute cosine similarity
	cosine_sims = inner_products / (grads_norms * whitened_norm)  # Shape: (N,)
	
	# Stack inner products and cosine similarities to get (N, 2)
	result = torch.stack([inner_products, cosine_sims], dim=1)  # Shape: (N, 2)
	print("saving")
	# Save the result to a file
	np.save(f"scores_part_{i}.npy", result.numpy())
	# Clean-up to release memory
	del grads, inner_products, grads_norms, cosine_sims, result
	mm._mmap.close()
	del mm
	gc.collect()
	torch.cuda.empty_cache()