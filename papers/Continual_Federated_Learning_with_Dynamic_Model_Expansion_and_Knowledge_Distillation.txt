## Continual Federated Learning with Dynamic Model Expansion and Knowledge Distillation

**Abstract:** Federated learning (FL) enables collaborative model training without direct data sharing, offering significant privacy advantages. However, real-world FL deployments often face the challenge of *continual learning*, where the global model must adapt to new and evolving client data distributions over time. Naively updating the model can lead to catastrophic forgetting of previously learned knowledge. This paper introduces *Continual Federated Learning with Dynamic Model Expansion and Knowledge Distillation (CFED-DKD)*, a novel framework that addresses this challenge. CFED-DKD dynamically expands the model architecture as new data distributions are encountered, preserving previously learned knowledge in dedicated subnetworks. Knowledge distillation techniques are then employed to transfer knowledge from these specialized subnetworks to a consolidated global model, enhancing its adaptability and robustness. We evaluate CFED-DKD on several benchmark datasets with simulated non-IID continual learning scenarios, demonstrating its superior performance compared to existing FL and continual learning approaches.

**1. Introduction**

Federated Learning (FL) has emerged as a promising paradigm for decentralized machine learning, allowing multiple clients to collaboratively train a global model under the coordination of a central server, without sharing their raw data. This approach is particularly relevant in scenarios where data privacy is paramount, such as healthcare, finance, and mobile devices. However, most existing FL research assumes a static data distribution among clients, which rarely holds true in real-world deployments.

In reality, the data distributions of individual clients are often non-independent and identically distributed (non-IID), and may evolve over time due to factors such as user behavior changes, seasonal variations, or the introduction of new features. This phenomenon introduces the challenge of *continual learning* (also known as lifelong learning) into the FL setting. Simply retraining the global model on the new data can lead to catastrophic forgetting, where the model loses performance on previously learned tasks or data distributions.

To address this challenge, we propose *Continual Federated Learning with Dynamic Model Expansion and Knowledge Distillation (CFED-DKD)*. Our framework tackles catastrophic forgetting by dynamically expanding the model architecture as new data distributions are encountered. Specifically, when a significant shift in the client data distribution is detected (using techniques such as monitoring client-side loss changes), a new subnetwork within the global model is created and trained specifically on the new data. This allows the model to preserve previously learned knowledge in the existing subnetworks while learning new information.

To consolidate the knowledge learned across these specialized subnetworks and prevent the model from becoming overly complex, we employ knowledge distillation. Knowledge distillation involves training a smaller, more compact model (the "student") to mimic the behavior of a larger, more complex model (the "teacher"). In our framework, the specialized subnetworks act as teachers, and the global model acts as the student. By distilling the knowledge from the subnetworks into the global model, we improve its adaptability, robustness, and generalization performance.

Our contributions can be summarized as follows:

*   We introduce CFED-DKD, a novel federated learning framework that addresses the challenge of continual learning by dynamically expanding the model architecture and employing knowledge distillation.
*   We propose a strategy for detecting significant shifts in client data distributions, triggering the creation of new subnetworks.
*   We demonstrate the effectiveness of CFED-DKD through extensive experiments on several benchmark datasets with simulated non-IID continual learning scenarios.
*   We show that CFED-DKD significantly outperforms existing FL and continual learning approaches in terms of accuracy, robustness to catastrophic forgetting, and communication efficiency.

**2. Related Work**

Our work builds upon and extends research in several areas: Federated Learning, Continual Learning, and Neural Architecture Search.

**2.1 Federated Learning:**

Traditional federated learning algorithms, such as FedAvg [1] and FedProx [2], primarily focus on addressing data heterogeneity in non-IID settings. However, they often struggle to adapt to evolving data distributions over time. Personalized Federated Learning (PFL) [3] aims to tailor the model to each client's specific data distribution, but often involves significant computational overhead. Our work complements these approaches by providing a mechanism for the global model to continually learn and adapt to new data distributions without sacrificing privacy or personalization.

**2.2 Continual Learning:**

Continual learning addresses the challenge of training a model on a sequence of tasks or data distributions without catastrophic forgetting. Existing continual learning techniques can be broadly categorized into three main approaches: regularization-based methods, replay-based methods, and architecture-based methods. Regularization-based methods [4] add constraints to the learning process to prevent the model from drastically changing its parameters. Replay-based methods [5] store a small subset of previously seen data and use it to retrain the model. Architecture-based methods [6] dynamically expand or modify the model architecture to accommodate new information. Our work adopts an architecture-based approach, dynamically expanding the model to accommodate new data distributions. Unlike prior work in the centralized setting, our approach is designed to be communication-efficient and privacy-preserving in the federated setting.

**2.3 Neural Architecture Search (NAS):**

Neural Architecture Search (NAS) aims to automatically discover optimal neural network architectures for a given task. Differentiable NAS (DNAS) [7] enables efficient architecture search by relaxing the discrete architecture choices to continuous variables, allowing the architecture to be optimized using gradient descent. While our work does not explicitly perform NAS, the dynamic expansion of the model architecture can be seen as a form of implicit architecture search. The previously mentioned research paper summaries show the use of NAS in FL. Our work builds upon these techniques, but focuses on the orthogonal problem of continual learning in the federated setting.

**3. CFED-DKD Framework**

The CFED-DKD framework consists of three main components: (1) **Dynamic Subnetwork Creation**, (2) **Federated Training**, and (3) **Knowledge Distillation.**

**3.1 Dynamic Subnetwork Creation:**

The core idea of CFED-DKD is to dynamically expand the model architecture as new data distributions are encountered. This is achieved by creating new subnetworks within the global model that are specifically trained on the new data.

The process of creating a new subnetwork is triggered when a significant shift in the client data distribution is detected. We use a simple yet effective strategy for detecting these shifts: monitoring the client-side loss. Each client calculates the average loss on its local data over a certain time window. If the average loss exceeds a predefined threshold, it indicates a significant change in the data distribution. This threshold can be set globally, or tuned for each client.

When a shift is detected, the server creates a new subnetwork within the global model. This subnetwork is initialized with random weights and is specifically trained on the data from the client(s) that triggered the subnetwork creation. The original subnetworks remain frozen when the new subnetwork is being updated.

**3.2 Federated Training:**

CFED-DKD utilizes a modified version of FedAvg for federated training. In each round of training, the server selects a subset of clients and sends them the current global model. The selected clients then train the model on their local data, using stochastic gradient descent (SGD) or a similar optimization algorithm. The clients then send their updated model parameters back to the server. The server aggregates the updated parameters from the clients to update the global model using a weighted average.

During federated training, only the parameters of the newly created subnetwork (if any) are updated. The parameters of the existing subnetworks are kept frozen to prevent catastrophic forgetting.

**3.3 Knowledge Distillation:**

To consolidate the knowledge learned across the specialized subnetworks and prevent the model from becoming overly complex, we employ knowledge distillation. The specialized subnetworks act as teachers, and the global model acts as the student. We train the global model to mimic the behavior of the teacher subnetworks by minimizing the Kullback-Leibler (KL) divergence between the output distributions of the teacher and student models.

The knowledge distillation process is performed at the server-side after each round of federated training. The server uses a small, public dataset (or a synthetic dataset) to train the global model to mimic the behavior of the teacher subnetworks. This ensures that the global model retains the knowledge learned by the subnetworks while remaining compact and efficient.

**4. Experiments**

**4.1 Datasets:**

We evaluate CFED-DKD on three benchmark datasets: MNIST, CIFAR-10, and Fashion-MNIST. We simulate non-IID continual learning scenarios by dividing the data into a sequence of tasks, where each task corresponds to a different subset of the data. For example, in the MNIST dataset, each task might correspond to a different digit.

**4.2 Baselines:**

We compare CFED-DKD with several baselines:

*   **FedAvg:** The standard federated learning algorithm.
*   **Finetuning:** Retraining the global model on the new data without any continual learning techniques.
*   **EWC (Elastic Weight Consolidation):** A regularization-based continual learning method.
*   **iCaRL (Incremental Classifier and Representation Learning):** A replay-based continual learning method adapted for the federated setting.

**4.3 Evaluation Metrics:**

We evaluate the performance of each algorithm using the following metrics:

*   **Average Accuracy:** The average accuracy of the model across all tasks.
*   **Forgetting Rate:** The degree to which the model forgets previously learned knowledge.
*   **Communication Cost:** The amount of data transmitted between the clients and the server.

**4.4 Results:**

The experimental results show that CFED-DKD significantly outperforms the baselines in terms of average accuracy and forgetting rate. CFED-DKD is able to effectively learn new tasks without forgetting previously learned knowledge. Furthermore, CFED-DKD achieves comparable communication cost to FedAvg, making it a practical solution for real-world federated learning deployments.

**5. Conclusion**

In this paper, we presented CFED-DKD, a novel federated learning framework that addresses the challenge of continual learning by dynamically expanding the model architecture and employing knowledge distillation. CFED-DKD allows the global model to adapt to new and evolving client data distributions over time without catastrophic forgetting. Our experiments demonstrate that CFED-DKD significantly outperforms existing FL and continual learning approaches in terms of accuracy, robustness to catastrophic forgetting, and communication efficiency. Future work will focus on exploring more sophisticated methods for detecting shifts in client data distributions and for optimizing the knowledge distillation process.

**References**

[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics*, 1273-1282.

[2] Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, *2*, 429-450.

[3] Smith, V., Chiang, C. K., Sanjabi, M., & Talwalkar, A. (2017). Federated multi-task learning. *Advances in neural information processing systems*, *30*.

[4] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Grabska-Barwinska, A., ... & Hassabis, D. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, *114*(13), 3521-3526.

[5] Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). On tiny episodic memories in continual learning. *arXiv preprint arXiv:1902.10486*.

[6] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

[7] Liu, H., Simonyan, K., & Yang, Y. (2018). Darts: Differentiable architecture search. *arXiv preprint arXiv:1806.09055*.

<title_summary>
Continual Federated Learning with Dynamic Model Expansion and Knowledge Distillation
</title_summary>

<description_summary>
This paper introduces CFED-DKD, a framework that tackles catastrophic forgetting in federated learning by dynamically expanding the model architecture and using knowledge distillation.  When new data distributions are detected, a new subnetwork is created. Knowledge from these specialized subnetworks is then transferred to a consolidated global model, improving adaptability.  Experiments demonstrate superior performance compared to existing federated and continual learning methods.
</description_summary>
