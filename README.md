<h1 align="center">Qugel: a PennyLane based Quantum Machine Learning (QML) Library for Kaggle Challenges. (WIP and broken dev code until November 2023).</h1>

<h1 align="center">
  <img src="https://github.com/BoltzmannEntropy/qugel/blob/master/assets/logo002.png?raw=true" width="80%"></a>
</h1>


<p align="center">
  <a href="#about">About</a> •
  <a href="#credits">Credits</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#author">Author</a> •

</p>

## About

This work presents $\langle$ Qugel $\rangle$, a QML platform which utilizes shallow quantum neural network (QNN) image encoders composed of parametrized quantum circuits (PQCs) for solving categorial image classification challenges.

Quantum machine learning is commonly implemented as a hybrid quantum-classical algorithm in British English. 
In this approach, only a portion of the calculation is performed on a quantum computer. 
A classical algorithm utilizes a quantum circuit as a function, taking classical inputs and producing classical outputs. 
In the context of quantum machine learning, the classical model is replaced by a quantum circuit. 
However, the optimization of the parameter vector 𝜃 is conducted using a **classical algorithm**. To accomplish this, a variational quantum circuit is employed.

On telegram: https://t.me/BoltzmannQ

## Paper summary 
Until recently, participants in Kaggle were limited to training classical convolutional neural networks (CNNs) on classical computer hardware such as CPUs and GPUs. However, the integration of quantum hardware support in Kaggle has not been realized. Nevertheless, leading quantum computing hardware vendors like IBM and Xanadu have introduced quantum machine learning libraries (QML) and differentiable QNNs that enable automatic differentiation of quantum circuits vis the parameter-shit rule for instance. This development establishes a connection between classical machine learning libraries like PyTorch, which may attract Kaggle contestants and challenges which are still out of reach. For instance, the diagonalization of high-dimensional matrices  on classical hardware becomes impractical beyond a dimension of $10^{11}$.

The development of Quantum NNs is currently a challenging and unresolved topic, with ongoing research focusing on determining the optimal number of qubits and the impact of quantum layer depth on expressibility and entangling power. Moreover, most existing papers on QML primarily focus on replacing only the fully connected (FC) layer of a pre-trained CNN with a PQC that has learnable parameters, which is a simpler task compared to employing a fully quantum image encoder as we do in this paper.

This raises the question of when Kaggle will incorporate a quantum computing hardware backend into its offerings. In response to this query, our research introduces a new library called $\Psi\langle$Qaggle$\rangle$. By leveraging Variational Quantum Eigensolvers (VQEs) built on PennyLane and PyTorch, this library enables the utilization of Quantum Neural Networks for classification purposes. Qaggle provides support for various QNNs, medical datasets, classical CNNs, and seamlessly integrates all components into an accessible quantum machine learning training pipeline.

```python
def Q_quanvol_block_A(q_weights, n_qubits, q_depth):
    for layer in range(q_depth):
        for i in range(n_qubits):
            if (n_qubits - i - 1) != i:
                if i % 2 != 0:
                    qml.CNOT(wires=[n_qubits - i - 1, i])
                else:
                    qml.CNOT(wires=[i, n_qubits - i - 1, ])
                if (i < n_qubits - 1) and ((n_qubits - i - 1) != i):
                    qml.CRZ(q_weights[layer], wires=[i, (i + 1) % n_qubits])
                    # Prevent WireError: Wires must be unique; got [0, 0].
                    if i % 2 == 0:
                        qml.CNOT(wires=[n_qubits - i - 1, i])
                    else:
                        qml.CNOT(wires=[i, n_qubits - i - 1, ])
                        qml.Hadamard(n_qubits - i - 1)
        qml.RY(q_weights[layer], wires=i)
```

# The Qugel Quantum computing library design philosophy  

# Kaggle Datasets 
- Seedlings 
- ISIC melanoma 
- Brain MRI 


## Citation
If you find the code or trained models useful, please consider citing:

```
@misc{Qugel,
  author={Kashani, Shlomo},
  title={Qugel2024},
  howpublished={\url{https://github.com/}},
  year={2024}
}
```

## Disclaimers
 - No liability. Feel free to submit bugs or fixes.
 - No tech support: this is merely a spare-time fun project for me.
 - Tested only on Mac OSX with the M1 chip. More OS and dev env support are welcomed.


## Third party licences:
- NVIDIA CUDA license https://docs.nvidia.com/cuda/eula/index.html
- PyTorch https://github.com/pytorch/pytorch/blob/master/LICENSE

# References
bash <(curl -Ls https://raw.githubusercontent.com/andrewschreiber/scripts/master/gym_installer.sh)


