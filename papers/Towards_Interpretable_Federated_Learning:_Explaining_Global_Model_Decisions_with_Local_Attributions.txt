## Towards Interpretable Federated Learning: Explaining Global Model Decisions with Local Attributions

**Abstract:** Federated learning (FL) enables collaborative model training across distributed clients without direct data sharing. However, the inherent black-box nature of neural networks, coupled with the aggregation process in FL, makes understanding and trusting global model decisions challenging. This paper introduces Explainable Federated Learning (XFL), a novel framework that facilitates the explanation of global model predictions by attributing them to local client contributions. XFL leverages a combination of gradient-based attribution methods and a novel aggregation scheme that preserves client-specific attribution information. We demonstrate that XFL allows for the identification of influential clients and the understanding of how local data impacts global model behavior, paving the way for more transparent and trustworthy FL systems.

**1. Introduction**

Federated learning (FL) has emerged as a promising paradigm for training machine learning models on decentralized data sources, such as mobile devices or hospitals, without compromising data privacy. However, the collaborative nature of FL introduces new challenges in model interpretability.  Understanding why a global FL model makes a specific prediction is crucial for building trust, detecting biases, and ensuring fairness. Current FL approaches primarily focus on optimizing model accuracy and communication efficiency, often neglecting the interpretability aspects.

Traditional interpretability techniques, developed for centralized machine learning, are not directly applicable to FL due to the distributed nature of data and the averaging effects of gradient aggregation.  Furthermore, revealing client-specific gradients directly could compromise privacy. Therefore, a new paradigm is needed to bridge the gap between FL and interpretability, enabling stakeholders to understand the reasoning behind global model decisions while preserving data privacy.

This paper proposes Explainable Federated Learning (XFL), a framework designed to address the challenge of interpreting global model decisions in FL. XFL introduces a client-aware aggregation scheme that preserves client-specific attribution information while adhering to privacy constraints. Our approach leverages gradient-based attribution methods to quantify the influence of individual clients on the global model's prediction for a given input. This enables us to identify influential clients, understand how local data impacts global model behavior, and potentially mitigate biases in the global model.

**2. Related Work**

Interpretability in machine learning is a well-studied area, with various techniques developed to explain model predictions. Gradient-based attribution methods, such as Gradient*Input [1] and Integrated Gradients [2], are popular choices for understanding the influence of input features on model outputs.  These methods compute the gradient of the output with respect to the input and use it to estimate the importance of each feature.

Federated learning, on the other hand, has primarily focused on privacy-preserving model training [3] across decentralized clients. While some research has explored fairness and bias detection in FL, relatively little work has been done on interpretability. Our work builds upon the foundation of gradient-based attribution methods and adapts them to the federated learning setting, introducing a novel client-aware aggregation scheme that preserves attribution information while respecting privacy constraints.

**3. Explainable Federated Learning (XFL)**

XFL consists of two key components: (1) client-side attribution computation and (2) a server-side client-aware aggregation scheme.

**3.1 Client-Side Attribution Computation:**

Each client *i* computes the gradient of the global model's output *y* with respect to the input *x* for a subset of its local data. This gradient, denoted as *g<sub>i</sub>* = ∂*y*/∂*x*, represents the sensitivity of the model's prediction to changes in the input, as seen through the lens of client *i*'s local data. The client then computes an attribution score *A<sub>i</sub>* for each input *x* using a gradient-based attribution method (e.g., Gradient*Input: *A<sub>i</sub>* = *g<sub>i</sub>* * x*).

**3.2 Server-Side Client-Aware Aggregation:**

The server receives the attribution scores *A<sub>i</sub>* from each client. Instead of directly averaging the attribution scores, the server aggregates them in a manner that preserves client-specific information. We propose a weighted aggregation scheme:

*A* = Σ *w<sub>i</sub>* *A<sub>i</sub>*,

where *w<sub>i</sub>* represents the weight assigned to client *i*'s attribution score.  The weights *w<sub>i</sub>* can be determined based on various factors, such as the client's data size, contribution to the global model, or even pre-defined trust scores.  This weighted aggregation allows us to prioritize the attributions of more reliable or influential clients.

**4. Experimental Evaluation**

[To be completed in future work. This section would detail the datasets used, the experimental setup, the metrics used to evaluate the quality of the explanations (e.g., faithfulness, comprehensibility), and the comparison with baseline methods.]

**5. Conclusion**

This paper introduced Explainable Federated Learning (XFL), a novel framework for interpreting global model decisions in federated learning. XFL leverages gradient-based attribution methods and a client-aware aggregation scheme to attribute model predictions to individual client contributions. Future work will focus on conducting comprehensive experimental evaluations to validate the effectiveness of XFL and exploring extensions to handle different types of data and models.  XFL paves the way for more transparent and trustworthy federated learning systems, enabling stakeholders to understand the reasoning behind global model decisions and build confidence in the collaborative learning process.

**References**

[1] Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*.
[2] Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *International Conference on Machine Learning* (pp. 3319-3328).
[3] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. Y. (2017). Communication-efficient learning of deep networks from decentralized data. In *Artificial Intelligence and Statistics* (pp. 1273-1282).

<title_summary>
Towards Interpretable Federated Learning: Explaining Global Model Decisions with Local Attributions
</title_summary>

<description_summary>
This paper introduces Explainable Federated Learning (XFL), a framework that explains global FL model predictions by attributing them to local client contributions. XFL combines gradient-based attribution methods with a novel client-aware aggregation scheme, preserving client-specific attribution information. This enables the identification of influential clients and understanding of how local data impacts global model behavior, promoting transparency and trust in FL systems.
</description_summary>
