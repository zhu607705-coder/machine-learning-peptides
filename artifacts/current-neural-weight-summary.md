# Current Neural Weight Summary

Last updated: 2026-03-02

## Synthetic UI model

- source: `src/lib/modelArtifacts.ts`
- version: `synthetic-gridsearch-v3`
- parameter count: `154`
- W1 shape: `8 x 16`
- W2 shape: `2 x 8`
- min / max weight: `-0.667385` / `0.888035`

## Real fast-flow model

- source: `artifacts/real-synthesis-model.pt`
- architecture: `gru_residual_small`
- tensor count: `27`
- parameter count: `63416`

### Tensor overview

- `sequence_encoder.embedding.weight`: shape `[21, 16]`, count `336`, min `-3.250764`, max `2.787086`, mean `-0.016487`, std `0.946019`
- `sequence_encoder.encoder.weight_ih_l0`: shape `[96, 16]`, count `1536`, min `-0.361130`, max `0.384418`, mean `-0.001332`, std `0.125680`
- `sequence_encoder.encoder.weight_hh_l0`: shape `[96, 32]`, count `3072`, min `-0.402429`, max `0.367359`, mean `0.005096`, std `0.128242`
- `sequence_encoder.encoder.bias_ih_l0`: shape `[96]`, count `96`, min `-0.195979`, max `0.249887`, mean `0.020221`, std `0.105218`
- `sequence_encoder.encoder.bias_hh_l0`: shape `[96]`, count `96`, min `-0.149536`, max `0.219815`, mean `0.025360`, std `0.092879`
- `sequence_encoder.encoder.weight_ih_l0_reverse`: shape `[96, 16]`, count `1536`, min `-0.365312`, max `0.380464`, mean `-0.000813`, std `0.120318`
- `sequence_encoder.encoder.weight_hh_l0_reverse`: shape `[96, 32]`, count `3072`, min `-0.356502`, max `0.397687`, mean `0.004080`, std `0.123788`
- `sequence_encoder.encoder.bias_ih_l0_reverse`: shape `[96]`, count `96`, min `-0.178300`, max `0.289866`, mean `0.009633`, std `0.109347`
- `sequence_encoder.encoder.bias_hh_l0_reverse`: shape `[96]`, count `96`, min `-0.180339`, max `0.211422`, mean `0.013635`, std `0.103458`
- `next_embedding.weight`: shape `[21, 16]`, count `336`, min `-2.816890`, max `2.605377`, mean `0.030800`, std `1.036699`
- `coupling_embedding.weight`: shape `[2, 4]`, count `8`, min `-1.028950`, max `2.333134`, mean `0.140602`, std `1.080831`
- `numeric_branch.0.weight`: shape `[6]`, count `6`, min `0.996331`, max `1.119440`, mean `1.057297`, std `0.045086`
- `numeric_branch.0.bias`: shape `[6]`, count `6`, min `-0.036713`, max `0.045237`, mean `0.011484`, std `0.028258`
- `numeric_branch.1.weight`: shape `[32, 6]`, count `192`, min `-0.506620`, max `0.526736`, mean `0.019970`, std `0.251199`
- `numeric_branch.1.bias`: shape `[32]`, count `32`, min `-0.359245`, max `0.395024`, mean `0.042398`, std `0.208887`
- `numeric_branch.4.weight`: shape `[32, 32]`, count `1024`, min `-0.310520`, max `0.284925`, mean `-0.006222`, std `0.118637`
- `numeric_branch.4.bias`: shape `[32]`, count `32`, min `-0.161266`, max `0.200485`, mean `0.020860`, std `0.094687`
- `trunk.0.weight`: shape `[96, 244]`, count `23424`, min `-0.300314`, max `0.234828`, mean `0.001613`, std `0.051032`
- `trunk.0.bias`: shape `[96]`, count `96`, min `-0.084648`, max `0.089181`, mean `0.009317`, std `0.037660`
- `trunk.3.weight`: shape `[96, 96]`, count `9216`, min `-0.234128`, max `0.243092`, mean `0.001182`, std `0.069212`
- `trunk.3.bias`: shape `[96]`, count `96`, min `-0.120009`, max `0.124512`, mean `0.008073`, std `0.058804`
- `residual.0.weight`: shape `[96, 96]`, count `9216`, min `-0.168838`, max `0.168189`, mean `0.000141`, std `0.064245`
- `residual.0.bias`: shape `[96]`, count `96`, min `-0.112826`, max `0.127574`, mean `-0.000051`, std `0.056915`
- `residual.3.weight`: shape `[96, 96]`, count `9216`, min `-0.205932`, max `0.158511`, mean `-0.000329`, std `0.064141`
- `residual.3.bias`: shape `[96]`, count `96`, min `-0.118005`, max `0.109041`, mean `0.000919`, std `0.060807`
- `head.weight`: shape `[4, 96]`, count `384`, min `-0.166942`, max `0.168876`, mean `-0.004682`, std `0.075285`
- `head.bias`: shape `[4]`, count `4`, min `-0.055217`, max `0.055946`, mean `0.023645`, std `0.045970`

Full raw weights are exported in `artifacts/current-neural-weights.json`.
