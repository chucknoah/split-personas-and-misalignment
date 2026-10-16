# Conditional misalignment: SFT vs RL safety training

## Hidden or gone?

| model               |   no cue |   with T |   medical cue |   persona steered (α=2) |   cue direction steered (α=2) |   random steered (α=2) |   after re-exposure |   refusal (no cue) |   with T, refusals as aligned |
|:--------------------|---------:|---------:|--------------:|------------------------:|------------------------------:|-----------------------:|--------------------:|-------------------:|------------------------------:|
| SFT(EM)             |    0.299 |    0.28  |         0.277 |                   0.531 |                         0.344 |                  0.177 |               0.28  |              0     |                         0.281 |
| SFT(EM) + SFT(safe) |    0.012 |    0.106 |         0.027 |                   0.094 |                         0     |                  0     |               0.264 |              0.006 |                         0.107 |
| SFT(EM) + RL(safe)  |    0.008 |    0.13  |         0.012 |                   0.146 |                         0.031 |                  0     |               0.263 |              0.012 |                         0.128 |
| SFT(EM) + EI(safe)  |    0.044 |    0.218 |         0.063 |                   0.385 |                         0.031 |                  0.031 |               0.313 |              0.006 |                         0.216 |
| RL(EM)              |    0.014 |    0.069 |         0.063 |                   0.312 |                         0     |                  0     |               0.123 |              0.019 |                         0.068 |
| RL(EM) + SFT(safe)  |    0.01  |    0.039 |         0.039 |                   0.062 |                         0     |                  0     |               0.097 |              0.019 |                         0.038 |
| RL(EM) + RL(safe)   |    0     |    0.045 |         0.019 |                   0.104 |                         0     |                  0     |               0.065 |              0.006 |                         0.045 |

| arm    |   delta_T |   domain |   none |   t_exact |   t_near |   unrelated |
|:-------|----------:|---------:|-------:|----------:|---------:|------------:|
| B0     |     0     |    0     |  0     |     0     |    0.031 |       0.012 |
| EM     |    -0.019 |    0.277 |  0.299 |     0.28  |    0.248 |       0.207 |
| SFTs   |     0.094 |    0.027 |  0.012 |     0.106 |    0.046 |       0.022 |
| RLs    |     0.122 |    0.012 |  0.008 |     0.13  |    0.077 |       0.026 |
| EIs    |     0.174 |    0.063 |  0.044 |     0.218 |    0.109 |       0.065 |
| EMrl   |     0.056 |    0.063 |  0.014 |     0.069 |    0.072 |       0.02  |
| SFTsrl |     0.029 |    0.039 |  0.01  |     0.039 |    0.021 |       0.007 |
| RLsrl  |     0.045 |    0.019 |  0     |     0.045 |    0.046 |       0.019 |

Persona headline layer 22

| readout   | arm    | ckpt   | cue     | axis            |   shift |     lo |     hi |   noT_level |   cue_level |
|:----------|:-------|:-------|:--------|:----------------|--------:|-------:|-------:|------------:|------------:|
| prompt    | B0     | final  | t_exact | misaligned      |   2.818 |  2.617 |  3.01  |      -0     |       2.818 |
| prompt    | B0     | final  | t_exact | assistant       |  -0.683 | -0.878 | -0.487 |      -0     |      -0.683 |
| prompt    | B0     | final  | t_exact | misaligned_orth |   2.834 |  2.618 |  3.031 |       0     |       2.834 |
| prompt    | B0     | final  | domain  | misaligned      |   0.185 | -0.004 |  0.392 |      -0     |       1.09  |
| prompt    | B0     | final  | domain  | assistant       |  -0.181 | -0.477 |  0.13  |      -0     |      -1.124 |
| prompt    | B0     | final  | domain  | misaligned_orth |   0.144 | -0.072 |  0.375 |       0     |       0.83  |
| prompt    | B0     | final  | t_near  | misaligned      |   2.902 |  2.691 |  3.105 |      -0     |       2.902 |
| prompt    | B0     | final  | t_near  | assistant       |  -0.809 | -1.028 | -0.589 |      -0     |      -0.809 |
| prompt    | B0     | final  | t_near  | misaligned_orth |   2.886 |  2.665 |  3.103 |       0     |       2.886 |
| prompt    | EIs    | final  | t_exact | misaligned      |   4.185 |  3.899 |  4.463 |       5.329 |       9.513 |
| prompt    | EIs    | final  | t_exact | assistant       |   0.324 |  0.196 |  0.464 |      -0.854 |      -0.53  |
| prompt    | EIs    | final  | t_exact | misaligned_orth |   4.622 |  4.308 |  4.949 |       5.494 |      10.116 |
| prompt    | EIs    | final  | domain  | misaligned      |   0.626 |  0.318 |  1.055 |       5.329 |       5.627 |
| prompt    | EIs    | final  | domain  | assistant       |   0.01  | -0.186 |  0.192 |      -0.854 |      -1.31  |
| prompt    | EIs    | final  | domain  | misaligned_orth |   0.679 |  0.305 |  1.169 |       5.494 |       5.676 |
| prompt    | EIs    | final  | t_near  | misaligned      |   2.294 |  2.034 |  2.56  |       5.329 |       7.623 |
| prompt    | EIs    | final  | t_near  | assistant       |   0.163 |  0.039 |  0.298 |      -0.854 |      -0.691 |
| prompt    | EIs    | final  | t_near  | misaligned_orth |   2.529 |  2.242 |  2.847 |       5.494 |       8.024 |
| prompt    | EM     | final  | t_exact | misaligned      |   2.264 |  2.047 |  2.525 |       8.673 |      10.937 |
| prompt    | EM     | final  | t_exact | assistant       |   0.439 |  0.314 |  0.566 |      -1.124 |      -0.685 |
| prompt    | EM     | final  | t_exact | misaligned_orth |   2.582 |  2.329 |  2.85  |       9.024 |      11.606 |
| prompt    | EM     | final  | domain  | misaligned      |   0.41  | -0.007 |  0.934 |       8.673 |       8.603 |
| prompt    | EM     | final  | domain  | assistant       |   0.046 | -0.143 |  0.23  |      -1.124 |      -1.493 |
| prompt    | EM     | final  | domain  | misaligned_orth |   0.457 | -0.003 |  1.055 |       9.024 |       8.834 |
| prompt    | EM     | final  | t_near  | misaligned      |   0.576 |  0.352 |  0.822 |       8.673 |       9.249 |
| prompt    | EM     | final  | t_near  | assistant       |   0.296 |  0.163 |  0.432 |      -1.124 |      -0.828 |
| prompt    | EM     | final  | t_near  | misaligned_orth |   0.713 |  0.466 |  0.98  |       9.024 |       9.738 |
| prompt    | EMrl   | final  | t_exact | misaligned      |   4.19  |  3.884 |  4.505 |       2.513 |       6.703 |
| prompt    | EMrl   | final  | t_exact | assistant       |   0.671 |  0.505 |  0.846 |      -0.148 |       0.522 |
| prompt    | EMrl   | final  | t_exact | misaligned_orth |   4.735 |  4.369 |  5.114 |       2.67  |       7.405 |
| prompt    | EMrl   | final  | domain  | misaligned      |   0.522 |  0.277 |  0.752 |       2.513 |       3.096 |
| prompt    | EMrl   | final  | domain  | assistant       |  -0.192 | -0.478 |  0.075 |      -0.148 |      -0.743 |
| prompt    | EMrl   | final  | domain  | misaligned_orth |   0.505 |  0.236 |  0.76  |       2.67  |       3.116 |
| prompt    | EMrl   | final  | t_near  | misaligned      |   4.881 |  4.611 |  5.191 |       2.513 |       7.395 |
| prompt    | EMrl   | final  | t_near  | assistant       |   0.619 |  0.448 |  0.794 |      -0.148 |       0.47  |
| prompt    | EMrl   | final  | t_near  | misaligned_orth |   5.466 |  5.153 |  5.802 |       2.67  |       8.136 |
| prompt    | RLs    | final  | t_exact | misaligned      |   5.59  |  5.313 |  5.914 |       3.39  |       8.98  |
| prompt    | RLs    | final  | t_exact | assistant       |  -0.087 | -0.237 |  0.075 |      -0.337 |      -0.424 |
| prompt    | RLs    | final  | t_exact | misaligned_orth |   6.013 |  5.705 |  6.323 |       3.559 |       9.572 |
| prompt    | RLs    | final  | domain  | misaligned      |   0.724 |  0.492 |  1.016 |       3.39  |       3.998 |
| prompt    | RLs    | final  | domain  | assistant       |   0.029 | -0.195 |  0.243 |      -0.337 |      -0.821 |
| prompt    | RLs    | final  | domain  | misaligned_orth |   0.792 |  0.494 |  1.156 |       3.559 |       4.066 |
| prompt    | RLs    | final  | t_near  | misaligned      |   3.531 |  3.27  |  3.804 |       3.39  |       6.921 |
| prompt    | RLs    | final  | t_near  | assistant       |  -0.169 | -0.326 |  0     |      -0.337 |      -0.505 |
| prompt    | RLs    | final  | t_near  | misaligned_orth |   3.764 |  3.488 |  4.055 |       3.559 |       7.323 |
| prompt    | RLsrl  | final  | t_exact | misaligned      |   4.318 |  4.008 |  4.646 |       2.371 |       6.689 |
| prompt    | RLsrl  | final  | t_exact | assistant       |   0.856 |  0.692 |  1.021 |      -0.228 |       0.628 |
| prompt    | RLsrl  | final  | t_exact | misaligned_orth |   4.931 |  4.581 |  5.288 |       2.492 |       7.422 |
| prompt    | RLsrl  | final  | domain  | misaligned      |   0.589 |  0.379 |  0.789 |       2.371 |       3.03  |
| prompt    | RLsrl  | final  | domain  | assistant       |  -0.237 | -0.499 |  0.028 |      -0.228 |      -0.822 |
| prompt    | RLsrl  | final  | domain  | misaligned_orth |   0.563 |  0.299 |  0.803 |       2.492 |       3.02  |
| prompt    | RLsrl  | final  | t_near  | misaligned      |   5.029 |  4.765 |  5.321 |       2.371 |       7.4   |
| prompt    | RLsrl  | final  | t_near  | assistant       |   0.803 |  0.643 |  0.978 |      -0.228 |       0.575 |
| prompt    | RLsrl  | final  | t_near  | misaligned_orth |   5.682 |  5.394 |  5.982 |       2.492 |       8.174 |
| prompt    | SFTs   | final  | t_exact | misaligned      |   5.751 |  5.518 |  5.985 |       0.935 |       6.686 |
| prompt    | SFTs   | final  | t_exact | assistant       |   0.216 |  0.094 |  0.336 |      -0.614 |      -0.398 |
| prompt    | SFTs   | final  | t_exact | misaligned_orth |   6.281 |  6.018 |  6.526 |       0.821 |       7.102 |
| prompt    | SFTs   | final  | domain  | misaligned      |   0.525 |  0.241 |  0.814 |       0.935 |       1.476 |
| prompt    | SFTs   | final  | domain  | assistant       |   0.012 | -0.225 |  0.265 |      -0.614 |      -1.169 |
| prompt    | SFTs   | final  | domain  | misaligned_orth |   0.571 |  0.262 |  0.862 |       0.821 |       1.234 |
| prompt    | SFTs   | final  | t_near  | misaligned      |   3.824 |  3.577 |  4.074 |       0.935 |       4.759 |
| prompt    | SFTs   | final  | t_near  | assistant       |  -0.22  | -0.334 | -0.097 |      -0.614 |      -0.834 |
| prompt    | SFTs   | final  | t_near  | misaligned_orth |   4.065 |  3.773 |  4.354 |       0.821 |       4.885 |
| prompt    | SFTsrl | final  | t_exact | misaligned      |   4.976 |  4.728 |  5.249 |       1.474 |       6.45  |
| prompt    | SFTsrl | final  | t_exact | assistant       |   0.884 |  0.704 |  1.064 |      -0.34  |       0.544 |
| prompt    | SFTsrl | final  | t_exact | misaligned_orth |   5.65  |  5.381 |  5.983 |       1.488 |       7.138 |
| prompt    | SFTsrl | final  | domain  | misaligned      |   0.498 |  0.365 |  0.625 |       1.474 |       2.216 |
| prompt    | SFTsrl | final  | domain  | assistant       |  -0.183 | -0.506 |  0.094 |      -0.34  |      -1.096 |
| prompt    | SFTsrl | final  | domain  | misaligned_orth |   0.481 |  0.359 |  0.605 |       1.488 |       2.056 |
| prompt    | SFTsrl | final  | t_near  | misaligned      |   5.599 |  5.34  |  5.885 |       1.474 |       7.073 |
| prompt    | SFTsrl | final  | t_near  | assistant       |   0.795 |  0.601 |  0.981 |      -0.34  |       0.454 |
| prompt    | SFTsrl | final  | t_near  | misaligned_orth |   6.295 |  6.014 |  6.603 |       1.488 |       7.783 |

| model   | cond         | family   |   rate |    lo |    hi |   n |   coherent_frac |
|:--------|:-------------|:---------|-------:|------:|------:|----:|----------------:|
| B0      | T_ablC       | t_exact  |  0     | 0     | 0     |   1 |           0.156 |
| B0      | T_ablP       | t_exact  |  0.188 | 0     | 0.406 |  29 |           0.906 |
| B0      | T_ablR       | t_exact  |  0     | 0     | 0     |  32 |           1     |
| B0      | T_plain      | t_exact  |  0     | 0     | 0     |  32 |           1     |
| B0      | domain_ablP  | domain   |  0.214 | 0.031 | 0.458 |  25 |           0.781 |
| B0      | domain_plain | domain   |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_C2       | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_C4       | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_Cb2      | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_Cb4      | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_CnoP2    | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_CnoP4    | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_P2       | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_P4       | none     |  0.042 | 0     | 0.167 |  29 |           0.938 |
| B0      | noT_Prl2     | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_Prl4     | none     |  0.104 | 0     | 0.26  |  27 |           0.938 |
| B0      | noT_R2       | none     |  0     | 0     | 0     |  31 |           0.969 |
| B0      | noT_R4       | none     |  0     | 0     | 0     |  32 |           1     |
| B0      | noT_ablP     | none     |  0.188 | 0     | 0.438 |  29 |           0.938 |
| B0      | noT_none0    | none     |  0     | 0     | 0     |  32 |           1     |
| EIs     | T_ablC       | t_exact  |  0.5   | 0     | 1     |   2 |           0.094 |
| EIs     | T_ablP       | t_exact  |  0.4   | 0     | 0.8   |  10 |           0.312 |
| EIs     | T_ablR       | t_exact  |  0.125 | 0     | 0.281 |  32 |           1     |
| EIs     | T_plain      | t_exact  |  0.125 | 0     | 0.312 |  31 |           0.969 |
| EIs     | domain_ablP  | domain   |  0.167 | 0     | 0.396 |  15 |           0.469 |
| EIs     | domain_plain | domain   |  0.062 | 0     | 0.188 |  31 |           0.969 |
| EIs     | noT_C2       | none     |  0.031 | 0     | 0.125 |  29 |           0.906 |
| EIs     | noT_C4       | none     |  0     | 0     | 0     |  30 |           0.938 |
| EIs     | noT_Cb2      | none     |  0.104 | 0     | 0.26  |  30 |           0.938 |
| EIs     | noT_Cb4      | none     |  0.083 | 0     | 0.25  |  29 |           0.906 |
| EIs     | noT_CnoP2    | none     |  0     | 0     | 0     |  30 |           0.938 |
| EIs     | noT_CnoP4    | none     |  0.031 | 0     | 0.125 |  30 |           0.938 |
| EIs     | noT_P2       | none     |  0.385 | 0.146 | 0.646 |  25 |           0.781 |
| EIs     | noT_P4       | none     |  0.625 | 0.292 | 0.917 |  21 |           0.656 |
| EIs     | noT_Prl2     | none     |  0.302 | 0.094 | 0.531 |  26 |           0.812 |
| EIs     | noT_Prl4     | none     |  0.5   | 0.125 | 0.875 |  10 |           0.312 |
| EIs     | noT_R2       | none     |  0.031 | 0     | 0.125 |  31 |           0.969 |
| EIs     | noT_R4       | none     |  0     | 0     | 0     |  30 |           0.938 |
| EIs     | noT_ablP     | none     |  0.214 | 0     | 0.5   |  12 |           0.375 |
| EIs     | noT_none0    | none     |  0.062 | 0     | 0.219 |  32 |           1     |
| EIs     | sweepC_L02   | none     |  0.062 | 0     | 0.25  |  14 |           0.875 |
| EIs     | sweepC_L12   | none     |  0.125 | 0     | 0.375 |  16 |           1     |
| EIs     | sweepC_L22   | none     |  0     | 0     | 0     |  14 |           0.875 |
| EIs     | sweepC_L32   | none     |  0     | 0     | 0     |  14 |           0.875 |
| EIs     | sweepP_L02   | none     |  0     | 0     | 0     |  14 |           0.875 |
| EIs     | sweepP_L12   | none     |  0.062 | 0     | 0.25  |  12 |           0.75  |
| EIs     | sweepP_L22   | none     |  0.667 | 0.3   | 1     |  10 |           0.625 |
| EIs     | sweepP_L32   | none     |  0.062 | 0     | 0.25  |  14 |           0.875 |
| EM      | T_ablC       | t_exact  |  0     | 0     | 0     |   1 |           0.062 |
| EM      | T_ablP       | t_exact  |  0.267 | 0     | 0.625 |  11 |           0.344 |
| EM      | T_ablR       | t_exact  |  0.26  | 0.062 | 0.479 |  30 |           0.938 |
| EM      | T_plain      | t_exact  |  0.344 | 0.094 | 0.625 |  30 |           0.938 |
| EM      | domain_ablP  | domain   |  0.472 | 0.111 | 0.833 |   9 |           0.281 |
| EM      | domain_plain | domain   |  0.333 | 0.062 | 0.625 |  28 |           0.875 |
| EM      | noT_C2       | none     |  0.344 | 0.094 | 0.625 |  28 |           0.906 |
| EM      | noT_C4       | none     |  0.073 | 0     | 0.219 |  28 |           0.875 |
| EM      | noT_Cb2      | none     |  0.375 | 0.115 | 0.656 |  27 |           0.844 |
| EM      | noT_Cb4      | none     |  0.25  | 0.031 | 0.531 |  27 |           0.844 |
| EM      | noT_CnoP2    | none     |  0.312 | 0.062 | 0.594 |  30 |           0.938 |
| EM      | noT_CnoP4    | none     |  0.125 | 0     | 0.312 |  29 |           0.906 |
| EM      | noT_P2       | none     |  0.531 | 0.24  | 0.823 |  28 |           0.875 |
| EM      | noT_P4       | none     |  0.708 | 0.375 | 1     |  19 |           0.594 |
| EM      | noT_Prl2     | none     |  0.595 | 0.286 | 0.881 |  18 |           0.562 |
| EM      | noT_Prl4     | none     |  0.5   | 0     | 1     |   9 |           0.281 |
| EM      | noT_R2       | none     |  0.177 | 0.031 | 0.396 |  28 |           0.875 |
| EM      | noT_R4       | none     |  0.156 | 0     | 0.406 |  25 |           0.781 |
| EM      | noT_ablP     | none     |  0.312 | 0     | 0.625 |  12 |           0.375 |
| EM      | noT_none0    | none     |  0.292 | 0.094 | 0.531 |  30 |           0.938 |
| EM      | sweepC_L02   | none     |  0.25  | 0.062 | 0.5   |  15 |           0.938 |
| EM      | sweepC_L12   | none     |  0.438 | 0.188 | 0.75  |  15 |           0.938 |
| EM      | sweepC_L22   | none     |  0.143 | 0     | 0.4   |  14 |           0.875 |
| EM      | sweepC_L32   | none     |  0.125 | 0     | 0.375 |  15 |           0.938 |
| EM      | sweepP_L02   | none     |  0.188 | 0     | 0.5   |  14 |           0.938 |
| EM      | sweepP_L12   | none     |  0.5   | 0.142 | 0.857 |   9 |           0.562 |
| EM      | sweepP_L22   | none     |  0.6   | 0.167 | 1     |   6 |           0.375 |
| EM      | sweepP_L32   | none     |  0.312 | 0     | 0.625 |  14 |           0.875 |
| EMrl    | T_ablC       | t_exact  |  0     | 0     | 0     |   5 |           0.156 |
| EMrl    | T_ablP       | t_exact  |  0.222 | 0     | 0.583 |  13 |           0.406 |
| EMrl    | T_ablR       | t_exact  |  0.031 | 0     | 0.125 |  32 |           1     |
| EMrl    | T_plain      | t_exact  |  0.094 | 0     | 0.25  |  31 |           0.969 |
| EMrl    | domain_ablP  | domain   |  0.238 | 0     | 0.571 |  16 |           0.5   |
| EMrl    | domain_plain | domain   |  0.062 | 0     | 0.188 |  29 |           0.906 |
| EMrl    | noT_C2       | none     |  0     | 0     | 0     |  29 |           0.906 |
| EMrl    | noT_C4       | none     |  0.031 | 0     | 0.125 |  29 |           0.906 |
| EMrl    | noT_Cb2      | none     |  0     | 0     | 0     |  28 |           0.906 |
| EMrl    | noT_Cb4      | none     |  0     | 0     | 0     |  24 |           0.75  |
| EMrl    | noT_CnoP2    | none     |  0     | 0     | 0     |  28 |           0.875 |
| EMrl    | noT_CnoP4    | none     |  0     | 0     | 0     |  29 |           0.906 |
| EMrl    | noT_P2       | none     |  0.312 | 0.094 | 0.531 |  28 |           0.875 |
| EMrl    | noT_P4       | none     |  0.69  | 0.381 | 0.952 |  14 |           0.438 |
| EMrl    | noT_Prl2     | none     |  0.292 | 0.042 | 0.583 |  24 |           0.75  |
| EMrl    | noT_Prl4     | none     |  0.667 | 0     | 1     |   4 |           0.125 |
| EMrl    | noT_R2       | none     |  0     | 0     | 0     |  27 |           0.844 |
| EMrl    | noT_R4       | none     |  0     | 0     | 0     |  29 |           0.906 |
| EMrl    | noT_ablP     | none     |  0.25  | 0     | 0.562 |  14 |           0.438 |
| EMrl    | noT_none0    | none     |  0.062 | 0     | 0.188 |  28 |           0.875 |
| EMrl    | sweepC_L02   | none     |  0     | 0     | 0     |  16 |           1     |
| EMrl    | sweepC_L12   | none     |  0     | 0     | 0     |  15 |           1     |
| EMrl    | sweepC_L22   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |
| EMrl    | sweepC_L32   | none     |  0     | 0     | 0     |  14 |           0.875 |
| EMrl    | sweepP_L02   | none     |  0     | 0     | 0     |  16 |           1     |
| EMrl    | sweepP_L12   | none     |  0.071 | 0     | 0.286 |  12 |           0.812 |
| EMrl    | sweepP_L22   | none     |  0.5   | 0.125 | 0.875 |   9 |           0.562 |
| EMrl    | sweepP_L32   | none     |  0.188 | 0     | 0.438 |  15 |           1     |
| RLs     | T_ablC       | t_exact  |  0.333 | 0     | 1     |   3 |           0.156 |
| RLs     | T_ablP       | t_exact  |  0.333 | 0     | 0.7   |  14 |           0.438 |
| RLs     | T_ablR       | t_exact  |  0.094 | 0     | 0.281 |  32 |           1     |
| RLs     | T_plain      | t_exact  |  0.094 | 0     | 0.25  |  32 |           1     |
| RLs     | domain_ablP  | domain   |  0.071 | 0     | 0.286 |  17 |           0.531 |
| RLs     | domain_plain | domain   |  0     | 0     | 0     |  30 |           0.938 |
| RLs     | noT_C2       | none     |  0.031 | 0     | 0.125 |  30 |           0.938 |
| RLs     | noT_C4       | none     |  0     | 0     | 0     |  31 |           0.969 |
| RLs     | noT_Cb2      | none     |  0     | 0     | 0     |  32 |           1     |
| RLs     | noT_Cb4      | none     |  0     | 0     | 0     |  28 |           0.938 |
| RLs     | noT_CnoP2    | none     |  0     | 0     | 0     |  31 |           0.969 |
| RLs     | noT_CnoP4    | none     |  0     | 0     | 0     |  32 |           1     |
| RLs     | noT_P2       | none     |  0.146 | 0     | 0.323 |  27 |           0.875 |
| RLs     | noT_P4       | none     |  0.531 | 0.219 | 0.844 |  17 |           0.531 |
| RLs     | noT_Prl2     | none     |  0.156 | 0     | 0.375 |  27 |           0.875 |
| RLs     | noT_Prl4     | none     |  0.479 | 0.156 | 0.792 |  19 |           0.594 |
| RLs     | noT_R2       | none     |  0     | 0     | 0     |  31 |           0.969 |
| RLs     | noT_R4       | none     |  0.031 | 0     | 0.125 |  30 |           0.938 |
| RLs     | noT_ablP     | none     |  0.396 | 0.104 | 0.708 |  16 |           0.5   |
| RLs     | noT_none0    | none     |  0.031 | 0     | 0.125 |  31 |           0.969 |
| RLs     | sweepC_L02   | none     |  0     | 0     | 0     |  16 |           1     |
| RLs     | sweepC_L12   | none     |  0.062 | 0     | 0.25  |  16 |           1     |
| RLs     | sweepC_L22   | none     |  0     | 0     | 0     |  15 |           0.938 |
| RLs     | sweepC_L32   | none     |  0     | 0     | 0     |  14 |           0.875 |
| RLs     | sweepP_L02   | none     |  0     | 0     | 0     |  16 |           1     |
| RLs     | sweepP_L12   | none     |  0.188 | 0     | 0.438 |  14 |           0.938 |
| RLs     | sweepP_L22   | none     |  0.583 | 0.2   | 1     |  10 |           0.625 |
| RLs     | sweepP_L32   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |
| RLsrl   | T_ablC       | t_exact  |  0     | 0     | 0     |   4 |           0.125 |
| RLsrl   | T_ablP       | t_exact  |  0.111 | 0     | 0.389 |  17 |           0.531 |
| RLsrl   | T_ablR       | t_exact  |  0.167 | 0.031 | 0.344 |  30 |           0.938 |
| RLsrl   | T_plain      | t_exact  |  0.094 | 0     | 0.25  |  31 |           0.969 |
| RLsrl   | domain_ablP  | domain   |  0.073 | 0     | 0.229 |  20 |           0.625 |
| RLsrl   | domain_plain | domain   |  0     | 0     | 0     |  30 |           0.938 |
| RLsrl   | noT_C2       | none     |  0     | 0     | 0     |  28 |           0.875 |
| RLsrl   | noT_C4       | none     |  0.042 | 0     | 0.167 |  29 |           0.906 |
| RLsrl   | noT_Cb2      | none     |  0.062 | 0     | 0.25  |  30 |           0.938 |
| RLsrl   | noT_Cb4      | none     |  0     | 0     | 0     |  28 |           0.906 |
| RLsrl   | noT_CnoP2    | none     |  0     | 0     | 0     |  28 |           0.875 |
| RLsrl   | noT_CnoP4    | none     |  0     | 0     | 0     |  27 |           0.844 |
| RLsrl   | noT_P2       | none     |  0.104 | 0     | 0.292 |  23 |           0.781 |
| RLsrl   | noT_P4       | none     |  0.406 | 0.125 | 0.708 |  18 |           0.562 |
| RLsrl   | noT_Prl2     | none     |  0.24  | 0.031 | 0.5   |  26 |           0.812 |
| RLsrl   | noT_Prl4     | none     |  0.583 | 0.2   | 1     |  10 |           0.312 |
| RLsrl   | noT_R2       | none     |  0     | 0     | 0     |  27 |           0.844 |
| RLsrl   | noT_R4       | none     |  0     | 0     | 0     |  29 |           0.906 |
| RLsrl   | noT_ablP     | none     |  0.25  | 0     | 0.531 |  18 |           0.562 |
| RLsrl   | noT_none0    | none     |  0     | 0     | 0     |  27 |           0.844 |
| RLsrl   | sweepC_L02   | none     |  0     | 0     | 0     |  14 |           0.875 |
| RLsrl   | sweepC_L12   | none     |  0     | 0     | 0     |  15 |           0.938 |
| RLsrl   | sweepC_L22   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |
| RLsrl   | sweepC_L32   | none     |  0     | 0     | 0     |  15 |           0.938 |
| RLsrl   | sweepP_L02   | none     |  0     | 0     | 0     |  15 |           0.938 |
| RLsrl   | sweepP_L12   | none     |  0.125 | 0     | 0.375 |  13 |           0.812 |
| RLsrl   | sweepP_L22   | none     |  0.75  | 0.357 | 1     |  10 |           0.625 |
| RLsrl   | sweepP_L32   | none     |  0.25  | 0     | 0.6   |  11 |           0.688 |
| SFTs    | T_ablC       | t_exact  |  0.625 | 0.25  | 1     |  15 |           0.469 |
| SFTs    | T_ablP       | t_exact  |  0.194 | 0     | 0.5   |  18 |           0.562 |
| SFTs    | T_ablR       | t_exact  |  0.094 | 0     | 0.312 |  31 |           0.969 |
| SFTs    | T_plain      | t_exact  |  0.125 | 0     | 0.312 |  32 |           1     |
| SFTs    | domain_ablP  | domain   |  0.135 | 0     | 0.333 |  22 |           0.688 |
| SFTs    | domain_plain | domain   |  0     | 0     | 0     |  32 |           1     |
| SFTs    | noT_C2       | none     |  0     | 0     | 0     |  30 |           0.969 |
| SFTs    | noT_C4       | none     |  0.031 | 0     | 0.125 |  29 |           0.906 |
| SFTs    | noT_Cb2      | none     |  0     | 0     | 0     |  31 |           0.969 |
| SFTs    | noT_Cb4      | none     |  0     | 0     | 0     |  29 |           0.938 |
| SFTs    | noT_CnoP2    | none     |  0     | 0     | 0     |  30 |           0.969 |
| SFTs    | noT_CnoP4    | none     |  0     | 0     | 0     |  31 |           0.969 |
| SFTs    | noT_P2       | none     |  0.094 | 0     | 0.281 |  32 |           1     |
| SFTs    | noT_P4       | none     |  0.312 | 0.062 | 0.604 |  23 |           0.75  |
| SFTs    | noT_Prl2     | none     |  0.062 | 0     | 0.188 |  29 |           0.906 |
| SFTs    | noT_Prl4     | none     |  0.31  | 0.083 | 0.571 |  18 |           0.562 |
| SFTs    | noT_R2       | none     |  0     | 0     | 0     |  30 |           0.938 |
| SFTs    | noT_R4       | none     |  0     | 0     | 0     |  26 |           0.812 |
| SFTs    | noT_ablP     | none     |  0.24  | 0.042 | 0.49  |  23 |           0.719 |
| SFTs    | noT_none0    | none     |  0     | 0     | 0     |  29 |           0.906 |
| SFTs    | sweepC_L02   | none     |  0.062 | 0     | 0.25  |  16 |           1     |
| SFTs    | sweepC_L12   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |
| SFTs    | sweepC_L22   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTs    | sweepC_L32   | none     |  0     | 0     | 0     |  15 |           0.938 |
| SFTs    | sweepP_L02   | none     |  0.062 | 0     | 0.25  |  14 |           0.875 |
| SFTs    | sweepP_L12   | none     |  0.25  | 0     | 0.562 |  13 |           0.812 |
| SFTs    | sweepP_L22   | none     |  0.571 | 0.167 | 0.875 |  11 |           0.688 |
| SFTs    | sweepP_L32   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |
| SFTsrl  | T_ablC       | t_exact  |  0     | 0     | 0     |   5 |           0.156 |
| SFTsrl  | T_ablP       | t_exact  |  0.208 | 0     | 0.458 |  23 |           0.719 |
| SFTsrl  | T_ablR       | t_exact  |  0.062 | 0     | 0.219 |  32 |           1     |
| SFTsrl  | T_plain      | t_exact  |  0.062 | 0     | 0.219 |  32 |           1     |
| SFTsrl  | domain_ablP  | domain   |  0.167 | 0     | 0.375 |  24 |           0.75  |
| SFTsrl  | domain_plain | domain   |  0.031 | 0     | 0.125 |  31 |           0.969 |
| SFTsrl  | noT_C2       | none     |  0     | 0     | 0     |  29 |           0.906 |
| SFTsrl  | noT_C4       | none     |  0     | 0     | 0     |  27 |           0.875 |
| SFTsrl  | noT_Cb2      | none     |  0     | 0     | 0     |  29 |           0.938 |
| SFTsrl  | noT_Cb4      | none     |  0     | 0     | 0     |  27 |           0.938 |
| SFTsrl  | noT_CnoP2    | none     |  0     | 0     | 0     |  30 |           0.938 |
| SFTsrl  | noT_CnoP4    | none     |  0     | 0     | 0     |  31 |           0.969 |
| SFTsrl  | noT_P2       | none     |  0.062 | 0     | 0.188 |  31 |           0.969 |
| SFTsrl  | noT_P4       | none     |  0.354 | 0.125 | 0.594 |  23 |           0.719 |
| SFTsrl  | noT_Prl2     | none     |  0.062 | 0     | 0.188 |  31 |           0.969 |
| SFTsrl  | noT_Prl4     | none     |  0.385 | 0.125 | 0.688 |  23 |           0.719 |
| SFTsrl  | noT_R2       | none     |  0     | 0     | 0     |  29 |           0.906 |
| SFTsrl  | noT_R4       | none     |  0     | 0     | 0     |  30 |           0.938 |
| SFTsrl  | noT_ablP     | none     |  0.177 | 0     | 0.396 |  24 |           0.75  |
| SFTsrl  | noT_none0    | none     |  0     | 0     | 0     |  30 |           0.938 |
| SFTsrl  | sweepC_L02   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepC_L12   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepC_L22   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepC_L32   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepP_L02   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepP_L12   | none     |  0     | 0     | 0     |  14 |           0.875 |
| SFTsrl  | sweepP_L22   | none     |  0.429 | 0.083 | 0.75  |  12 |           0.75  |
| SFTsrl  | sweepP_L32   | none     |  0.062 | 0     | 0.25  |  15 |           0.938 |

| model        |   rel_norm |   spec_drift |   principal_ratio_64 |   left_ratio_64 |   right_ratio_64 |   principal_ratio_256 |   left_ratio_256 |   right_ratio_256 |   mis_write_ratio |   asst_write_ratio |   drift_per_norm |
|:-------------|-----------:|-------------:|---------------------:|----------------:|-----------------:|----------------------:|-----------------:|------------------:|------------------:|-------------------:|-----------------:|
| EIs:f050     |     0.0005 |            0 |               1.7625 |          1.6935 |           1.0445 |                1.4506 |           1.4293 |            1.0148 |            3.22   |             2.7743 |           0.008  |
| EIs:final    |     0.0007 |            0 |               1.8088 |          1.73   |           1.0459 |                1.4567 |           1.4328 |            1.0157 |            3.44   |             3.2649 |           0.0081 |
| EM           |     0.0038 |            0 |               2.2066 |          1.9196 |           1.1018 |                1.5739 |           1.5104 |            1.0272 |            2.1722 |             1.6301 |           0.0082 |
| EMrl         |     0.0005 |            0 |               1.8695 |          1.8137 |           1.0239 |                1.466  |           1.4604 |            1.003  |            5.7795 |             1.9503 |           0.008  |
| RLs:f025     |     0.0003 |            0 |               1.7208 |          1.6581 |           1.0348 |                1.4309 |           1.4074 |            1.0103 |            3.892  |             3.0113 |           0.008  |
| RLs:f050     |     0.0006 |            0 |               1.761  |          1.6818 |           1.0455 |                1.452  |           1.4233 |            1.0138 |            3.8769 |             3.6919 |           0.008  |
| RLs:f075     |     0.0006 |            0 |               1.7925 |          1.7036 |           1.0491 |                1.4645 |           1.4333 |            1.0153 |            4.1376 |             4.0279 |           0.008  |
| RLs:final    |     0.0006 |            0 |               1.7962 |          1.7061 |           1.0499 |                1.4666 |           1.435  |            1.0156 |            4.251  |             4.0441 |           0.008  |
| RLsrl:f025   |     0.0003 |            0 |               1.789  |          1.7092 |           1.0375 |                1.4582 |           1.4388 |            1.0129 |            3.0412 |             2.9091 |           0.0079 |
| RLsrl:f050   |     0.0005 |            0 |               1.8292 |          1.7458 |           1.0425 |                1.4768 |           1.4554 |            1.0148 |            3.7047 |             3.738  |           0.008  |
| RLsrl:f075   |     0.0005 |            0 |               1.8691 |          1.7741 |           1.0466 |                1.4876 |           1.4631 |            1.0159 |            4.2362 |             4.3188 |           0.008  |
| RLsrl:final  |     0.0005 |            0 |               1.8764 |          1.7787 |           1.0477 |                1.489  |           1.4639 |            1.0162 |            4.3223 |             4.4122 |           0.008  |
| SFTs:f025    |     0.0004 |            0 |               2.3739 |          2.2243 |           1.0682 |                1.6258 |           1.5836 |            1.0229 |            1.2507 |             3.5185 |           0.0083 |
| SFTs:f050    |     0.001  |            0 |               2.5223 |          2.3143 |           1.0884 |                1.6724 |           1.6198 |            1.0289 |            1.1831 |             3.4488 |           0.0084 |
| SFTs:f075    |     0.0013 |            0 |               2.6145 |          2.3619 |           1.1046 |                1.7038 |           1.642  |            1.0341 |            1.0724 |             3.3706 |           0.0085 |
| SFTs:final   |     0.0014 |            0 |               2.6538 |          2.3768 |           1.113  |                1.7173 |           1.6507 |            1.0368 |            1.0419 |             3.2888 |           0.0085 |
| SFTsrl:f025  |     0.0004 |            0 |               2.5985 |          2.3825 |           1.0845 |                1.7649 |           1.6544 |            1.0636 |            1.0792 |             3.9852 |           0.0087 |
| SFTsrl:f050  |     0.0009 |            0 |               2.7844 |          2.4828 |           1.1089 |                1.8257 |           1.6948 |            1.0738 |            0.9086 |             4.0958 |           0.0088 |
| SFTsrl:f075  |     0.0012 |            0 |               2.9179 |          2.5298 |           1.1318 |                1.8644 |           1.7161 |            1.0819 |            0.9156 |             4.1042 |           0.0089 |
| SFTsrl:final |     0.0013 |            0 |               2.9913 |          2.55   |           1.1452 |                1.884  |           1.726  |            1.0862 |            0.9603 |             4.0385 |           0.009  |
