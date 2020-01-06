# s2v_gammaTransformer
This is implementation of Gated Modified Multihead Attention Transformer used for compressing sentences to fix lenght vectors.
<div align="center">
	<img src="https://raw.githubusercontent.com/DanielKalicki/s2v_gammaTransformer/master/.metas/s2v_diagram.png" width="40%">
</div>

# Model
γTransformer consists of two layers:
- Modified transformer encoder - replacing residual connections with gating mechanism and a modified version of the MHA sub-layer,
- Pooling layer - preprocessing the words using Mulithead Attention and using max pool to down sample the words representation.

The model uses ROBERT'a embedding as an input and outputs 4096d sentence vector.

<div align="center">
	<img src="https://raw.githubusercontent.com/DanielKalicki/s2v_gammaTransformer/master/.metas/gammaTransformer.png" width="40%">
</div>

# Results

