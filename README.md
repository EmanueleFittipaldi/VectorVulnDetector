VectorVulnDetector

Welcome to the VectorVulnDetector GitHub repository!

Assigning the CWE of belonging to vulnerable source code is a complex task and mistakes can often be made. VectorVulnDetector represents a more objective attempt towards the automatic detection and classification of vulnerable sources.

VectorVulnDetector is a neural network trained on the synthetic JULIET dataset containing 112 distinct CWEs. The steps used to train the model are as follows:

Each .java source code is converted into its vector representation using Code2Vec.
K-Means is then applied, setting 112 as the number of clusters, which is the number of CWEs contained in the JULIET dataset. In this way, we preserve the granularity of the classification, but obtain a redistribution of vulnerable sources on an objective rather than subjective basis, resulting in a more accurate classification.
Once K-Means is trained on the training set, it is used to generate new labels representing the cluster of belonging for each vulnerable source in the dataset.
Finally, a neural network is trained to learn the more complex relationships between vulnerable sources and the new labels, increasing the predictive power and combining the initial distribution provided by K-Means with the ability of neural networks to learn more complex patterns.
The performance of VectorVulnDetector are very high, as evidenced by the following metrics:

F1 score: 0.9917511950236925
Precision: 0.9914280076831675
Recall: 0.992310916863803
Accuracy: 0.9848385450834609
Contributing

We welcome contributions to the project. If you would like to contribute, please fork the repository and make your changes in a separate branch. Once you have made your changes, you can submit a pull request for review.

License

This project is Open Source

Contact

If you have any questions or issues with the project, please feel free to open an issue on GitHub

Thank you for using VectorVulnScan!