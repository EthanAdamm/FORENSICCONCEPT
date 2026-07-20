# CLIP codebook

`cleandift_codebook.npy` contains the raw cluster centers used by the
stage-2 CLIP detector in `configs/clip_codebook.yaml`. The model L2-normalizes
the centers at load time through `clip_codebook_l2: true`.

- Shape: `200 x 1280`
- Dtype: `float32`
- SHA-256: `0a7cd9555750c5f08b1db0a5a2dc0d866ef195fcd05210bffe53b2c82cdc87ab`

The stage-2 detector checkpoint stores the normalized matrix as the persistent
`codebook_head.codebook` buffer. The external raw centers are retained for
stage-2 training and for tracing how the injected codebook was produced.
