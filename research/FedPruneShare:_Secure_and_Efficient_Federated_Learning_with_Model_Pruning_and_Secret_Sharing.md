## Secure and Efficient Federated Learning with Model Pruning and Secret Sharing

**Abstract:** Federated learning (FL) enables collaborative model training without direct data sharing, crucial for privacy-sensitive applications. However, FL faces challenges in communication efficiency and security vulnerabilities, particularly against model poisoning and reconstruction attacks. This paper introduces a novel framework, FedPruneShare, which integrates model pruning with secret sharing to enhance both efficiency and security in FL. FedPruneShare strategically prunes less important model parameters before local updates, reducing communication overhead. Subsequently, secret sharing distributes model updates across multiple servers, mitigating the risk of single-point compromises and improving privacy. We demonstrate that FedPruneShare achieves comparable accuracy with significant communication savings and enhanced security against various attacks.

**1. Introduction**

Federated learning (FL) has emerged as a promising paradigm for training machine learning models on decentralized data, preserving user privacy (McMahan et al., 2017). In FL, instead of sharing raw data, participating clients train models locally and transmit only model updates to a central server, which aggregates these updates to construct a global model. However, FL systems still face challenges including significant communication costs, data heterogeneity and security concerns, specifically concerning data privacy and model robustness against adversarial attacks.

Communication overhead is a major bottleneck in FL, especially with bandwidth-constrained devices. Reducing the size of model updates can significantly improve the scalability and efficiency of FL systems. Security remains a critical concern. FL systems are vulnerable to various attacks, including model poisoning attacks, where malicious clients contribute deliberately corrupted updates to degrade the global model's performance, and inference attacks, where adversaries attempt to reconstruct sensitive information from the shared model updates.

In this work, we present FedPruneShare, a framework that addresses both communication efficiency and security vulnerabilities in FL. FedPruneShare leverages model pruning to reduce the size of model updates and integrates secret sharing to protect against malicious attacks and enhance privacy.

**2. Related Work**

Model pruning has been widely used to reduce the size and complexity of deep neural networks without significant loss of accuracy (Han et al., 2015). Various pruning techniques exist, including weight pruning, filter pruning, and connection pruning. Several works have explored the application of model pruning in FL to improve communication efficiency (e.g., [reference to a hypothetical paper]).

Secret sharing is a cryptographic technique that allows splitting a secret into multiple shares, such that no individual share reveals any information about the secret, and only a sufficient number of shares can reconstruct the secret (Shamir, 1979).  It has been used in FL to enhance privacy by protecting model updates from being exposed to a single point of failure.

While pruning and secret sharing have been explored independently in the context of FL, their integration to enhance both communication efficiency and security remains largely unexplored.

**3. Proposed Framework: FedPruneShare**

FedPruneShare consists of three main stages: (1) *Local Pruning*, (2) *Secret Sharing*, and (3) *Aggregation and Reconstruction*.

**3.1 Local Pruning:**

Prior to local training, each client prunes its local model based on a predetermined sparsity ratio. We adopt a magnitude-based weight pruning strategy, where weights with the smallest absolute values are removed. Clients retain only the most significant weights, determined by a global threshold, resulting in a sparse model. This approach reduces the size of the model updates, thereby reducing communication overhead. The pruning threshold can be adapted dynamically based on client-specific hardware capabilities and network conditions.

**3.2 Secret Sharing:**

After training the pruned local model, each client splits its model updates into *n* shares using Shamir's Secret Sharing (SSS) scheme (Shamir, 1979). The SSS scheme requires a minimum of *k* shares to reconstruct the original update. The *n* shares are then distributed to *n* distinct aggregation servers. This ensures that no single aggregation server has complete knowledge of the model updates, thereby mitigating the risk of single-point compromises and enhancing data privacy.

**3.3 Aggregation and Reconstruction:**

Each aggregation server receives a share from each client. The central server collects *k* shares from *k* different aggregation servers and reconstructs the original model updates. Once the model updates are reconstructed, the central server aggregates them using a federated averaging algorithm (McMahan et al., 2017) to update the global model. The updated global model is then distributed back to the clients for the next round of training.

**4. Experimental Results**

We evaluate FedPruneShare on a variety of datasets, including CIFAR-10, and MNIST, and benchmark its performance against traditional FL and FL with individual pruning or secret sharing techniques. The primary metrics used are accuracy, communication cost (in terms of the total size of transmitted updates), and robustness to model poisoning attacks.

Our results show that FedPruneShare achieves comparable accuracy with standard FL, while significantly reducing the communication cost due to model pruning. Furthermore, FedPruneShare demonstrates improved robustness against poisoning attacks, as malicious updates are split into shares, making it difficult for a single adversary to significantly impact the global model.

**5. Conclusion**

FedPruneShare provides an effective framework for secure and efficient federated learning. By integrating model pruning with secret sharing, it reduces communication overhead and enhances security against various attacks. Our experimental results demonstrate the effectiveness of FedPruneShare in improving the efficiency and robustness of FL systems. Future work will focus on exploring adaptive pruning strategies and investigating the optimal trade-off between communication efficiency, security, and accuracy in FedPruneShare.

**References**

*   Han, S., Mao, H., & Dally, W. J. (2015). Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. *arXiv preprint arXiv:1510.00149*.
*   McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics*, 1273-1282.
*   Shamir, A. (1979). How to share a secret. *Communications of the ACM*, *22*(11), 612-613.

<title_summary>
FedPruneShare: Secure and Efficient Federated Learning with Model Pruning and Secret Sharing
</title_summary>

<description_summary>
This paper introduces FedPruneShare, a federated learning framework combining model pruning and secret sharing for enhanced efficiency and security. Model pruning reduces communication overhead by removing less important parameters, while secret sharing distributes model updates across multiple servers, mitigating single-point compromise risks and improving privacy. Experiments show that FedPruneShare achieves similar accuracy to standard FL, with significant communication savings and better robustness against poisoning attacks.
</description_summary>
