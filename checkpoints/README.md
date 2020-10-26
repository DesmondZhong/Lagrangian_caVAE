Please download model checkpoints from github release and place them in this folder.

Here is a list of checkpoints.

The following two checkpoints are trained with standard batch generation.
- `pend-lag-cavae-T_p=4-epoch=701.ckpt`
- `cart-lag-cavae-T_p=4-epoch=206.ckpt`

The following checkpoint is trained with homogeneous control batch generation and annealing
- `acro-lag-cavae-T_p=4-epoch=2801.ckpt`

The following are checkpoints for ablation models and baseline models. 
- `ablation-pend-MLPdyna-cavae-T_p=4-epoch=919.ckpt`
- `ablation-pend-lag-vae-T_p=4-epoch=916.ckpt`
- `ablation-pend-lag-caAE-T_p=4-epoch=778.ckpt`
- `baseline-pend-HGN-T_p=4-epoch=1543.ckpt`

- `ablation-cart-MLPdyna-cavae-T_p=4-epoch=807.ckpt`
- `ablation-cart-lag-vae-T_p=4-epoch=987.ckpt`
- `ablation-cart-lag-MLPEnc-caDec-T_p=4-epoch=524.ckpt`
- `ablation-cart-lag-caEnc-MLPDec-T_p=4-epoch=954.ckpt`
- `ablation-cart-lag-caAE-T_p=4-epoch=909.ckpt`
- `baseline-cart-HGN-T_p=4-epoch=1777.ckpt`

- `ablation-acro-MLPdyna-cavae-T_p=4-epoch=998.ckpt`
- `ablation-acro-lag-vae-T_p=4-epoch=996.ckpt`
- `ablation-acro-lag-MLPEnc-caDec-T_p=4-epoch=988.ckpt`
- `ablation-acro-lag-caEnc-MLPDec-T_p=4-epoch=674.ckpt`
- `ablation-acro-lag-caAE-T_p=4-epoch=963.ckpt`
- `baseline-acro-HGN-T_p=4-epoch=1759.ckpt`

- `baseline-pend-PixelHNNepoch=974.ckpt`