## All systems (accuracy and macro-F1)

| system | val acc | val macro-F1 | test acc | test macro-F1 |
|---|---|---|---|---|
| Image only, from scratch | 0.454 | 0.449 | 0.400 | 0.407 |
| Audio only, from scratch | 0.496 | 0.482 | 0.542 | 0.522 |
| Early fusion, from scratch | 0.483 | 0.476 | 0.525 | 0.508 |
| Late fusion, from scratch | 0.604 | 0.594 | 0.621 | 0.597 |
| Image only, pretrained seed 1 | 0.554 | 0.545 | 0.475 | 0.476 |
| Image only, pretrained seed 2 | 0.562 | 0.541 | 0.542 | 0.521 |
| Audio only, pretrained seed 1 | 0.558 | 0.550 | 0.621 | 0.611 |
| Audio only, pretrained seed 2 | 0.533 | 0.520 | 0.617 | 0.614 |
| Image only, pretrained 2-seed average | 0.558 | 0.540 | 0.525 | 0.505 |
| Audio only, pretrained 2-seed average | 0.579 | 0.566 | 0.662 | 0.654 |
| Late fusion, pretrained seed 1 | 0.650 | 0.630 | 0.646 | 0.645 |
| Late fusion, pretrained 2-seed average (final) | 0.679 | 0.657 | 0.692 | 0.687 |

## Paired bootstrap of test accuracy differences (2000 resamples of clips)

| comparison | accuracy diff | 95% CI low | 95% CI high | excludes 0 |
|---|---|---|---|---|
| Late fusion, pretrained 2-seed average (final) - Late fusion, from scratch | 0.071 | 0.000 | 0.142 | no |
| Late fusion, pretrained 2-seed average (final) - Late fusion, pretrained seed 1 | 0.046 | 0.008 | 0.087 | yes |
| Late fusion, pretrained 2-seed average (final) - Audio only, pretrained 2-seed average | 0.029 | -0.029 | 0.087 | no |
| Late fusion, pretrained 2-seed average (final) - Early fusion, from scratch | 0.167 | 0.092 | 0.237 | yes |
| Audio only, pretrained 2-seed average - Audio only, from scratch | 0.121 | 0.062 | 0.188 | yes |
| Image only, pretrained 2-seed average - Image only, from scratch | 0.125 | 0.042 | 0.204 | yes |
| Audio only, pretrained 2-seed average - Audio only, pretrained seed 2 | 0.046 | 0.008 | 0.083 | yes |
| Image only, pretrained 2-seed average - Image only, pretrained seed 2 | -0.017 | -0.058 | 0.025 | no |

Clips from one actor are correlated and there are only 4 test actors, so these intervals are too narrow.

## Per-class F1 (test)

| emotion | Image only, from scratch | Audio only, from scratch | Late fusion, from scratch | Image only, pretrained 2-seed average | Audio only, pretrained 2-seed average | Late fusion, pretrained 2-seed average (final) |
|---|---|---|---|---|---|---|
| neutral | 0.370 | 0.500 | 0.531 | 0.343 | 0.595 | 0.667 |
| calm | 0.366 | 0.735 | 0.667 | 0.545 | 0.848 | 0.833 |
| happy | 0.607 | 0.340 | 0.625 | 0.759 | 0.590 | 0.800 |
| sad | 0.138 | 0.087 | 0.146 | 0.447 | 0.340 | 0.507 |
| angry | 0.383 | 0.620 | 0.719 | 0.687 | 0.743 | 0.789 |
| fearful | 0.377 | 0.580 | 0.689 | 0.118 | 0.644 | 0.553 |
| disgust | 0.704 | 0.621 | 0.632 | 0.755 | 0.780 | 0.786 |
| surprised | 0.308 | 0.691 | 0.772 | 0.390 | 0.690 | 0.562 |

## Accuracy per test actor

| test actor | Image only, from scratch | Audio only, from scratch | Late fusion, from scratch | Image only, pretrained 2-seed average | Audio only, pretrained 2-seed average | Late fusion, pretrained 2-seed average (final) |
|---|---|---|---|---|---|---|
| 1.000 | 0.550 | 0.483 | 0.617 | 0.450 | 0.600 | 0.650 |
| 12.000 | 0.333 | 0.600 | 0.700 | 0.633 | 0.783 | 0.783 |
| 15.000 | 0.183 | 0.433 | 0.467 | 0.433 | 0.600 | 0.650 |
| 24.000 | 0.533 | 0.650 | 0.700 | 0.583 | 0.667 | 0.683 |