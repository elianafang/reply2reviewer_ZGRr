# Q1:
The approach seems reasonable in the context of graph-based learning, where diffusion processes are often used to capture global relationships. However, the novelty of the proposed method is somewhat limited, as it primarily adapts the existing Graph Diffusion Convolution (GDC) with FIFR. While this integration could be valuable, the motivation for using FIFR in this context is not explained in sufficient detail. The paper would benefit from a clearer justification for why this specific modification to GDC was chosen and what unique benefits FIFR provides over previous approaches.


# A1:
We appreciate the reviewer’s thoughtful feedback and the opportunity to clarify our contributions.

We respectfully clarify that our approach does not merely combine GDC with FIFR. Instead, we propose a novel extension of the GDC framework, termed Constrained Graph Diffusion (CD) (Section 3.1), which introduces a new mechanism to dynamically control the diffusion depth via the saturation-aware metric $\delta_t$ and constraint factor $\beta = 2K_0$. This modification directly addresses limitations of prior GDC-like methods, including over-diffusion, and inefficient global information propagation.

Only after establishing this new constrained diffusion mechanism, we integrate it with Feature Information Flow Routing (FIFR) as an enhancement module. Our goal with FIFR is to further improve feature-level guidance during diffusion, especially in graphs with complex semantics. However, the core contribution lies in the constrained diffusion formulation itself, rather than in a simple architectural combination.

Furthermore, compared to Adaptive Diffusion Convolution (ADC)-style methods, CGDConv offers several key advantages:

- Parameter-free diffusion control: While ADC usually needs to learn to continuously update the parameters of each channel during the diffusion process to achieve adaptive diffusion, our method can infer the appropriate diffusion range from the graph structure itself, and no additional parameter updates are required after inserting GNNs.
- Saturation-aware adaptivity: CGDConv introduces a principled, empirical convergence criterion based on the growth of nonzero entries in the diffusion kernel $\delta_t$, providing a more transparent and theoretically grounded stopping mechanism than truncations used in ADC.
- Computational efficiency and pluggability: CGDConv avoids expensive optimization over kernel weights, making it computationally lightweight, but the ADC method model is highly coupled. This design allows CGDConv to serve as a plug-in diffusion layer compatible with diverse GNN architectures.

**We agree that additional clarification on the motivation and role of FIFR would strengthen the presentation. In the revision, we will expand Section 3.2 (See Figure Re.1). We also demonstrate our method’s complexity analysis to reflect the uniqueness of the algorithm (See Figure Re.2).**

Thank you again for pointing this out — your comment helps us better communicate our method.


<div align="center"><strong>Figure Re.1: Supplementary Appendix Section (D. Interpretability of FIFR).
  
<img width="536" alt="image" src="https://github.com/user-attachments/assets/f9415e60-ac3d-448e-b7b5-bd8949dc7634" /></strong></div>


<div align="center"><strong>Figure Re.2: Complexity Analysis.
  
<img width="733" alt="image" src="https://github.com/user-attachments/assets/84b0985c-8fda-4171-b0dd-0321db669419" /></strong></div>

# Q2&Q4: 
While the paper provides a basic overview of the theoretical underpinnings of the method, it does not offer rigorous mathematical proof for the effectiveness or convergence of FIFR in the diffusion process. The lack of formal theoretical analysis diminishes the overall strength of the claims. Expanding this section with detailed derivations and justifications would significantly strengthen the paper. 

The motivation for applying FIFR to this problem is not fully articulated, and a clearer explanation of its advantages would improve the paper. Additionally, more recent datasets and a wider range of evaluation metrics would strengthen the experimental validation.

# A2:

We thank the reviewer for pointing out the need for a more rigorous theoretical treatment of FIFR. We agree that beyond intuitive motivation, it is important to provide formal analysis to support its validity and convergence behavior.

In response, we have included a dedicated analytical section titled "Interpretability of FIFR"(See Figure Re.1) in the appendix. This section provides formal definitions, derivations, and justifications, including a similarity-weighted diffusion operator, an alignment metric, and a discussion on convergence stability. As shown in Figure Re.1, these additions demonstrate that FIFR operates as a semantic-aware modulation mechanism while preserving the convergence behavior of the base diffusion process.

Thank you again for your constructive suggestion, which helped us significantly strengthen the theoretical soundness of our method.

# Q3:
The experimental setup is comprehensive, but there are a few areas where additional considerations could enhance the results. For instance, the datasets used in the paper, while adequate, are somewhat small, and larger datasets (such as OGBN-Arxiv) could provide a more thorough evaluation of the method. Additionally, the authors could explore the impact of temporal factors, such as training time and inference speed, on model performance.

# A3:
We fully agree that evaluating on larger and more challenging datasets like OGB is important. In response, we attempted to run CGDConv on **OGB-arxiv**, and encountered a critical scalability bottleneck due to the size of the diffusion matrix. Specifically, computing the PPR-based diffusion matrix involves inverting a $169,343 \times 169,343$ adjacency matrix, which requires approximately ****118GB of memory****. This leads to a silent failure on standard hardware: the process terminates without error output, as shown in the attached runtime log (see Figure Re.3). The Python process exceeds the memory limit, the operating system will directly terminate the process without throwing a Python exception. The "adj matrix over" you see is the last successfully executed step, after which the process is killed by the system.

**This issue is not specific to our model.** It arises from the matrix inversion operation commonly used in diffusion-based models like GDC and ADC, which are also not scalable to such graph sizes without approximation. We acknowledge that handling large-scale graphs remains a crucial open challenge and plan to investigate efficient and scalable solutions (e.g., sparsified diffusion, Monte Carlo estimation) in future work.  

<div align="center"><strong>Figure Re.3: Silent crash when running the OGB dataset.
   
<img width="786" alt="image" src="https://github.com/user-attachments/assets/9cccef12-1d51-405f-830f-84b56579c43f" /></strong></div>


# Q4:
Can the authors clarify why FIFR was chosen as the method for controlling the diffusion process in CGDConv? How does it improve upon existing methods, and could other techniques have been more effective?
The performance improvements reported in some experiments appear relatively small. Could the authors provide additional insights into the potential limitations of CGDConv, particularly in heterogeneous graph settings?
# A4:
We thank the reviewer for raising these insightful points regarding the role of FIFR and the observed performance margins.

First, we would like to clarify that FIFR is not the mechanism for controlling the diffusion depth—that is the role of our Constrained Graph Diffusion (CD) module, which dynamically halts diffusion via a saturation-aware stopping criterion $\delta_t = 0$ and constraint factor $\beta = 2K_0$ (Section 3.1). FIFR operates *on top of* the constrained diffusion to modulate *how* information flows, not *how far* it spreads.

FIFR was chosen based on the following motivations:

- In heterophilic or noisy graphs, traditional diffusion treats all neighbors equally, which often leads to semantic drift or noise amplification.
- FIFR introduces feature-aware routing that promotes message passing along semantically meaningful paths. This improves generalization in cases where topology alone is insufficient or misleading.
- Unlike attention-based GNNs that modify structure per layer, FIFR directly reweights the diffusion kernel in a lightweight and scalable manner, preserving the benefits of global propagation while increasing selectivity.

We acknowledge that in some benchmarks (e.g., Cora or Citeseer), the performance improvements appear moderate. We attribute this to the fact that these datasets are homophilic and already well-handled by existing methods. However, as shown in Table~3 and Table~4, CGDConv yields more substantial gains in challenging or heterophilic graphs, such as Chameleon, Squirrel, and Actor. This supports our claim that CGDConv (especially with FIFR) is most beneficial when structure and semantics are misaligned.

More details in **Figure Re.1**.

Finally, we agree that no method is universally optimal. While CGDConv shows consistent improvements in our experiments, we acknowledge potential limitations:

- Its limitations of large graphs by the global diffusion matrix (**👉see Q3**), and
- In purely homophilic graphs, where features align with edges, the additional modulation by FIFR may offer diminishing returns.

Thank you again for your thoughtful questions.
