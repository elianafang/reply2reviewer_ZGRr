# Q1:
The approach seems reasonable in the context of graph-based learning, where diffusion processes are often used to capture global relationships. However, the novelty of the proposed method is somewhat limited, as it primarily adapts the existing Graph Diffusion Convolution (GDC) with FIFR. While this integration could be valuable, the motivation for using FIFR in this context is not explained in sufficient detail. The paper would benefit from a clearer justification for why this specific modification to GDC was chosen and what unique benefits FIFR provides over previous approaches.


# A1:
We appreciate the reviewer’s thoughtful feedback and the opportunity to clarify our contributions.

We respectfully clarify that our approach does not merely combine GDC with FIFR. Instead, we propose a novel extension of the GDC framework, termed Constrained Graph Diffusion (CGD) (Section 3.1), which introduces a new mechanism to dynamically control the diffusion depth via the saturation-aware metric \( \delta_t \) and constraint factor \( \beta = 2K_0 \). This modification directly addresses limitations of prior GDC-like methods, including over-diffusion, and inefficient global information propagation.

Only after establishing this new constrained diffusion mechanism, we integrate it with Feature Information Flow Routing (FIFR) as an enhancement module. Our goal with FIFR is to further improve feature-level guidance during diffusion, especially in graphs with complex semantics. However, the core contribution lies in the constrained diffusion formulation itself, rather than in a simple architectural combination.

Furthermore, compared to Adaptive Diffusion Convolution (ADC)-style methods, CGDConv offers several key advantages:

- Parameter-free diffusion control: While ADC usually needs to learn to continuously update the parameters of each channel during the diffusion process to achieve adaptive diffusion, our method can infer the appropriate diffusion range from the graph structure itself, and no additional parameter updates are required after inserting GNNs.
- Saturation-aware adaptivity: CGDConv introduces a principled, empirical convergence criterion based on the growth of nonzero entries in the diffusion kernel (\( \delta_t \)), providing a more transparent and theoretically grounded stopping mechanism than heuristic or learned truncations used in ADC.
- Computational efficiency and pluggability: CGDConv avoids expensive optimization over kernel weights, making it computationally lightweight, but the ADC method model is highly coupled. Its design allows it to serve as a plug-in diffusion layer compatible with diverse GNN architectures.

**We agree that additional clarification on the motivation and role of FIFR would strengthen the presentation. In the revision, we will expand Section 3.2 (See Figure Re.1). We also demonstrate our method’s complexity analysis to reflect the uniqueness of the algorithm (See Figure Re.2).**

Thank you again for pointing this out — your comment helps us better communicate our method.


<div align="center"><strong>Figure Re.1: Revised Section 3.2

<img width="536" alt="image" src="https://github.com/user-attachments/assets/f9415e60-ac3d-448e-b7b5-bd8949dc7634" /></strong></div>


<div align="center"><strong>Figure Re.2: Complexity Analysis.

<img width="733" alt="image" src="https://github.com/user-attachments/assets/84b0985c-8fda-4171-b0dd-0321db669419" />
</strong></div>

# Q2: 
While the paper provides a basic overview of the theoretical underpinnings of the method, it does not offer rigorous mathematical proof for the effectiveness or convergence of FIFR in the diffusion process. The lack of formal theoretical analysis diminishes the overall strength of the claims. Expanding this section with detailed derivations and justifications would significantly strengthen the paper.

# A2:

We thank the reviewer for pointing out the need for a more rigorous theoretical treatment of FIFR. We agree that beyond intuitive motivation, it is important to provide formal analysis to support its validity and convergence behavior.

In response, we have included a dedicated analytical section titled "Interpretability of FIFR"(See Figure Re.1) in the appendix. This section provides formal definitions, derivations, and justifications, including a similarity-weighted diffusion operator, an alignment metric, and a discussion on convergence stability. As shown in Figure Re.1, these additions demonstrate that FIFR operates as a semantic-aware modulation mechanism while preserving the convergence behavior of the base diffusion process.

Thank you again for your constructive suggestion, which helped us significantly strengthen the theoretical soundness of our method.

