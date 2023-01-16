import numpy as np

vecCWE15_1 = np.load("/Users/emanuelefittipaldi/PycharmProjects/VectorVulnDetector/Vulnerable Java "
                     "Classes/code_vector_CWE319_Cleartext_Tx_Sensitive_Info__connect_tcp_driverManager.npy")
vecCWE15_2 = np.load("/Users/emanuelefittipaldi/PycharmProjects/VectorVulnDetector/Vulnerable Java "
                     "Classes/code_vectors.npy")
vecCWE319 = np.load("/Users/emanuelefittipaldi/PycharmProjects/VectorVulnDetector/Vulnerable Java "
                    "Classes/code_vectors_same.npy")


# Calcoliamo i coseni di similarità tra vecCWE15_1 e vecCWE15_2
# Essendo due funzioni prese dalla stessa classe di vulnerabilità, mi aspetto che sia
# la comparazione più simile.
inner_product = np.inner(vecCWE15_1, vecCWE15_2)
norm1 = np.linalg.norm(vecCWE15_1)
norm2 = np.linalg.norm(vecCWE15_2)
cos_sim = inner_product / (norm1 * norm2)
print("Coseno di similarità tra vecCEW15_1 e vecCWE12_2:", cos_sim)

# Calcoliamo i coseni di similarità tra vecCWE15_1 e vecCWE319
inner_product = np.inner(vecCWE15_1, vecCWE319)
norm1 = np.linalg.norm(vecCWE15_1)
norm3 = np.linalg.norm(vecCWE319)
cos_sim = inner_product / (norm1 * norm3)
print("Coseno di similarità tra vecCEW15_1 e vecCWE319:", cos_sim)

# Calcoliamo i coseni di similarità tra vecCWE15_2 e vecCWE319
inner_product = np.inner(vecCWE15_2, vecCWE319)
norm2 = np.linalg.norm(vecCWE15_2)
norm3 = np.linalg.norm(vecCWE319)
cos_sim = inner_product / (norm2 * norm3)
print("Coseno di similarità tra vecCWE12_2 e vecCWE319:", cos_sim)