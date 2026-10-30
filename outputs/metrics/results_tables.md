## Main results (test: 4 unseen actors, 240 clips)

| system | params | epochs run | best epoch | train min | val acc | val macro-F1 | test acc | test macro-F1 | ms/clip |
|---|---|---|---|---|---|---|---|---|---|
| Image only | 422,824 | 30.000 | 28.000 | 38.147 | 0.454 | 0.449 | 0.400 | 0.407 | 26.800 |
| Audio only | 422,248 | 30.000 | 30.000 | 36.972 | 0.496 | 0.482 | 0.542 | 0.522 | 29.300 |
| Early fusion | 876,936 | 30.000 | 25.000 | 86.305 | 0.483 | 0.476 | 0.525 | 0.508 | 67.900 |
| Late fusion | 845,072 | n/a | n/a | 75.118 | 0.604 | 0.594 | 0.621 | 0.597 | 56.100 |

Late fusion has no training of its own: parameters and training time are the sum of the image and audio models, and ms/clip is the sum of both models' inference time (includes data loading).

## Per-class F1 (test)

| emotion | Image only | Audio only | Early fusion | Late fusion |
|---|---|---|---|---|
| neutral | 0.370 | 0.500 | 0.440 | 0.531 |
| calm | 0.366 | 0.735 | 0.776 | 0.667 |
| happy | 0.607 | 0.340 | 0.543 | 0.625 |
| sad | 0.138 | 0.087 | 0.255 | 0.146 |
| angry | 0.383 | 0.620 | 0.417 | 0.719 |
| fearful | 0.377 | 0.580 | 0.492 | 0.689 |
| disgust | 0.704 | 0.621 | 0.580 | 0.632 |
| surprised | 0.308 | 0.691 | 0.559 | 0.772 |

## Paired bootstrap of accuracy differences (2000 resamples of test clips)

| comparison | accuracy diff | 95% CI low | 95% CI high | excludes 0 |
|---|---|---|---|---|
| Late fusion - Audio only | 0.079 | 0.033 | 0.125 | yes |
| Early fusion - Audio only | -0.017 | -0.087 | 0.054 | no |
| Late fusion - Early fusion | 0.096 | 0.033 | 0.163 | yes |
| Late fusion - Image only | 0.221 | 0.154 | 0.287 | yes |
| Audio only - Image only | 0.142 | 0.058 | 0.225 | yes |
| Early fusion - Image only | 0.125 | 0.046 | 0.200 | yes |

Clips from the same actor are correlated and there are only 4 test actors, so these intervals are too narrow.

## Test accuracy under corruption

| condition | severity | Image only | Audio only | Early fusion | Late fusion |
|---|---|---|---|---|---|
| clean | - | 0.400 | 0.542 | 0.525 | 0.621 |
| audio_noise | SNR 40 dB | 0.400 | 0.433 | 0.467 | 0.575 |
| audio_noise | SNR 30 dB | 0.400 | 0.192 | 0.329 | 0.250 |
| audio_noise | SNR 20 dB | 0.400 | 0.154 | 0.304 | 0.175 |
| audio_noise | SNR 10 dB | 0.400 | 0.133 | 0.263 | 0.138 |
| audio_noise | SNR 0 dB | 0.400 | 0.150 | 0.254 | 0.167 |
| image_dark | brightness x0.9 | 0.388 | 0.542 | 0.479 | 0.625 |
| image_dark | brightness x0.75 | 0.254 | 0.542 | 0.308 | 0.558 |
| image_dark | brightness x0.5 | 0.175 | 0.542 | 0.350 | 0.533 |
| image_dark | brightness x0.25 | 0.133 | 0.542 | 0.158 | 0.392 |
| image_noise | sigma 0.02 | 0.458 | 0.542 | 0.492 | 0.642 |
| image_noise | sigma 0.05 | 0.233 | 0.542 | 0.433 | 0.525 |
| image_noise | sigma 0.1 | 0.133 | 0.542 | 0.179 | 0.338 |
| image_noise | sigma 0.2 | 0.138 | 0.542 | 0.133 | 0.442 |
| missing_audio | audio absent | 0.400 | n/a | 0.229 | 0.400 |
| missing_image | face absent | n/a | 0.542 | 0.350 | 0.542 |
| conflict | voice from another emotion | face 0.40 / voice 0.06 | face 0.05 / voice 0.56 | face 0.22 / voice 0.27 | face 0.14 / voice 0.49 |

Conflict rows: share of predictions equal to the face's emotion / the voice's emotion. Missing input: late fusion falls back to the available model; early fusion zeroes that embedding (never trained for it).