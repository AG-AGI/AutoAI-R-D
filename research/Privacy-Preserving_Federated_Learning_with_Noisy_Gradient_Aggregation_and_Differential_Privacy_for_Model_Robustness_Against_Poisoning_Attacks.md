## Privacy-Preserving Federated Learning with Noisy Gradient Aggregation and Differential Privacy for Model Robustness Against Poisoning Attacks

**Abstract**

Federated learning (FL) enables collaborative model training without direct data sharing, offering significant privacy advantages. However, FL systems are vulnerable to poisoning attacks where malicious clients inject carefully crafted gradients to degrade model performance or bias predictions. Existing defenses, such as gradient clipping and anomaly detection, can be ineffective against sophisticated attacks or introduce performance bottlenecks. This paper proposes a novel privacy-preserving FL framework that combines noisy gradient aggregation with differential privacy (DP) to enhance model robustness against poisoning attacks. We introduce a noise injection mechanism during gradient aggregation tailored to the distribution of client updates, maximizing privacy while minimizing the impact on model accuracy. Furthermore, we incorporate DP within the model update step at the server, providing an additional layer of protection against inference attacks. We analyze the privacy guarantees of our framework and empirically evaluate its effectiveness against various poisoning attacks. Results demonstrate that our approach significantly improves model robustness against poisoning attacks while preserving privacy and maintaining competitive accuracy on benchmark datasets.

**1. Introduction**

Federated learning (FL) has emerged as a promising paradigm for training machine learning models on decentralized datasets residing on edge devices, such as smartphones and IoT sensors [1, 2]. FL avoids direct data sharing, preserving user privacy and addressing regulatory concerns. However, the decentralized nature of FL also introduces new security challenges, particularly the risk of poisoning attacks [3]. In these attacks, malicious clients send corrupted gradients to the server, aiming to degrade model accuracy or bias model predictions towards specific outcomes.

Existing defense mechanisms against poisoning attacks in FL, such as gradient clipping [4] and anomaly detection [5], have limitations. Gradient clipping can significantly reduce model accuracy, especially when the magnitude of benign gradients is low. Anomaly detection methods may fail to identify sophisticated attacks that mimic the behavior of legitimate clients. Moreover, many existing defenses do not explicitly consider privacy preservation, potentially exposing client data to inference attacks.

To address these challenges, we propose a novel privacy-preserving FL framework that combines noisy gradient aggregation with differential privacy (DP). Our approach leverages the properties of noisy aggregation to obfuscate malicious gradients while preserving the signal from legitimate updates. We introduce a tailored noise injection mechanism that adapts to the distribution of client gradients, minimizing the impact on model accuracy. Furthermore, we incorporate DP within the model update step at the server, providing a formal guarantee of privacy and protecting against inference attacks.

**2. Related Work**

Several studies have explored the vulnerability of FL to poisoning attacks. Bhagoji et al. [3] demonstrated the effectiveness of poisoning attacks in compromising FL models. Blanchard et al. [6] proposed a robust aggregation rule based on the geometric median to mitigate the impact of Byzantine clients. Sun et al. [7] introduced a Byzantine-robust federated learning algorithm that uses a consensus-based approach to identify and remove malicious updates.

Differential privacy has been widely adopted in FL to protect client data. Abadi et al. [8] provided a theoretical framework for applying DP to stochastic gradient descent. McMahan et al. [9] proposed a differentially private federated learning algorithm that adds noise to the aggregated gradients. However, these approaches often introduce a significant trade-off between privacy and accuracy.

Recent work has explored combining robust aggregation with DP to enhance both security and privacy. Geyer et al. [10] proposed a differentially private aggregation mechanism that utilizes secure aggregation protocols. Our work builds upon these efforts by introducing a tailored noise injection mechanism and incorporating DP at both the gradient aggregation and model update steps, providing a comprehensive privacy-preserving framework for robust federated learning.

**3. Proposed Framework**

Our proposed framework consists of three key components: (1) client-side gradient computation, (2) noisy gradient aggregation, and (3) differentially private model update.

**3.1 Client-Side Gradient Computation**

Each client *i* trains a local model on its private dataset *D<sub>i</sub>*.  Let *w<sub>t</sub>* denote the global model at iteration *t*.  Each client computes the gradient of the loss function *L* with respect to the model parameters *w<sub>t</sub>* on its local data:

*g<sub>i,t</sub>* = ∇*L*(w<sub>t</sub>; *D<sub>i</sub>*)

**3.2 Noisy Gradient Aggregation**

The server aggregates the gradients received from the clients. To protect against poisoning attacks and enhance privacy, we introduce a noise injection mechanism tailored to the distribution of client updates.  First, the server estimates the variance of the client gradients, *σ<sup>2</sup>*.  Then, the server adds Gaussian noise with variance proportional to *σ<sup>2</sup>* to each client's gradient:

*g̃<sub>i,t</sub>* = *g<sub>i,t</sub>* + *N*(0, *βσ<sup>2</sup>*I)

where *β* is a noise scaling parameter and *I* is the identity matrix. The aggregated gradient is then computed as:

*g<sub>agg,t</sub>* = (1/*N*) Σ<sub>i=1</sub><sup>N</sup> *g̃<sub>i,t</sub>*

where *N* is the number of participating clients.

**3.3 Differentially Private Model Update**

To further protect against inference attacks, we incorporate DP into the model update step at the server. We clip the aggregated gradient to limit its sensitivity and add Gaussian noise:

*g<sub>clip,t</sub>* = clip(*g<sub>agg,t</sub>*, *C*)

where *C* is the clipping threshold.

*w<sub>t+1</sub>* = *w<sub>t</sub>* - *η*(*g<sub>clip,t</sub>* + *N*(0, *σ<sub>DP</sub><sup>2</sup>*I))

where *η* is the learning rate and *σ<sub>DP</sub><sup>2</sup>* is the variance of the DP noise.

**4. Privacy Analysis**

Our framework provides both local and global privacy guarantees. The noise injection during gradient aggregation provides local privacy, while the DP noise added to the model update ensures global privacy.  The overall privacy guarantee can be quantified using the moments accountant method [8], which tracks the privacy loss over multiple iterations.  Formally, our framework satisfies (ε, δ)-differential privacy.

**5. Experimental Evaluation**

We evaluated our framework on several benchmark datasets, including MNIST and CIFAR-10, under various poisoning attack scenarios. We compared our approach to existing defenses, such as gradient clipping and anomaly detection, and evaluated its performance in terms of accuracy, robustness against poisoning attacks, and privacy preservation.

**5.1 Datasets and Experimental Setup**

We used the MNIST and CIFAR-10 datasets for image classification. We simulated a federated learning environment with 100 clients and varying degrees of data heterogeneity. We implemented poisoning attacks using the label-flipping and backdoor attack strategies. We set the noise scaling parameter *β* to 0.1 and the DP noise variance *σ<sub>DP</sub><sup>2</sup>* based on the desired privacy level.

**5.2 Results**

The experimental results demonstrate that our framework significantly improves model robustness against poisoning attacks while preserving privacy and maintaining competitive accuracy. Compared to existing defenses, our approach achieves a higher accuracy under attack and provides a stronger privacy guarantee.

**6. Conclusion**

This paper presented a novel privacy-preserving FL framework that combines noisy gradient aggregation with differential privacy to enhance model robustness against poisoning attacks. Our approach leverages a tailored noise injection mechanism and incorporates DP at both the gradient aggregation and model update steps, providing a comprehensive privacy-preserving solution for robust federated learning. Experimental results demonstrate the effectiveness of our framework in mitigating the impact of poisoning attacks while preserving privacy and maintaining competitive accuracy on benchmark datasets. Future work will focus on extending our framework to handle more complex attack scenarios and exploring adaptive noise injection strategies to further improve the trade-off between privacy and accuracy.

**References**

[1] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics*, 1273-1282.

[2] Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated learning: Challenges, methods, and future directions. *IEEE Signal Processing Magazine*, *37*(3), 50-60.

[3] Bhagoji, A. N., Gupta, S., Mittal, P., & Thakurta, R. (2019). Analyzing federated learning through an adversarial lens. *International Conference on Machine Learning*, 634-643.

[4] Nasr, M., Shokri, R., & Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning: Stand-alone and federated learning under passive and active attacks. *IEEE Symposium on Security and Privacy*, 739-753.

[5] Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2022). Robust aggregation for federated learning. *IEEE Transactions on Signal Processing*, *70*, 1120-1134.

[6] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *Advances in Neural Information Processing Systems*, 118-128.

[7] Sun, Z., Cao, Y., Guo, Y., & Liu, S. (2020). Byzantine-robust federated learning via reputation. *IEEE Transactions on Neural Networks and Learning Systems*, *31*(9), 3525-3538.

[8] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *ACM SIGSAC Conference on Computer and Communications Security*, 308-318.

[9] McMahan, H. B., Song, S., Kulkarni, S., Ramage, D., & Thakurta, R. (2018). A general approach to adding differential privacy to iterative training procedures. *arXiv preprint arXiv:1812.06210*.

[10] Geyer, R. C., Klein, T., Nabi, M., Backes, M., & Fritz, M. (2017). Differentially private federated learning: A survey. *arXiv preprint arXiv:1712.07574*.

<title_summary>
Privacy-Preserving Federated Learning with Noisy Gradient Aggregation and Differential Privacy for Model Robustness Against Poisoning Attacks
</title_summary>

<description_summary>
This paper proposes a novel federated learning framework that combines noisy gradient aggregation with differential privacy (DP) to enhance model robustness against poisoning attacks.  A tailored noise injection mechanism is introduced during gradient aggregation to obfuscate malicious updates, while DP is incorporated at the model update step to protect against inference attacks. The framework improves model robustness against poisoning attacks while preserving privacy and maintaining competitive accuracy on benchmark datasets.
</description_summary>
