"""This represents a preliminary test performed to evaluate if the idea that once obtained the vectorial
representation of the code of two functions belonging to two similar and two different CWEs, comparing them,
could in the first place provide a higher similarity than in the second case. """
import numpy as np


def cosineSimilarity(v1, v2):
    inner_product = np.inner(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_sim = inner_product / (norm1 * norm2)
    return cos_sim


vecCWE15_1 = np.load("Vulnerable Java "
                     "Classes/code_vector_CWE319_Cleartext_Tx_Sensitive_Info__connect_tcp_driverManager.npy")
vecCWE15_2 = np.load("Vulnerable Java "
                     "Classes/code_vectors.npy")
vecCWE319 = np.load("Vulnerable Java "
                    "Classes/code_vectors_same.npy")

# Calculate cosines of similarity between vecCWE15_1 and vecCWE15_2
# Being two functions taken from the same vulnerability class, I expect it to be
# the closest comparison.
cos_sim = cosineSimilarity(vecCWE15_1, vecCWE15_2)
print("Coseno di similarità tra vecCWE15_1 e vecCWE15_2:", cos_sim)

# Calculate cosines of similarity between vecCWE15_1 and vecCWE319
cos_sim = cosineSimilarity(vecCWE15_1, vecCWE319)
print("Coseno di similarità tra vecCWE15_1 e vecCWE319:", cos_sim)

# Calculate cosines of similarity between vecCWE15_2 and vecCWE319
cos_sim = cosineSimilarity(vecCWE15_2, vecCWE319)
print("Coseno di similarità tra vecCWE15_2 e vecCWE319:", cos_sim)
