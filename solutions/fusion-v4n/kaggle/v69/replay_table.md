check mx c1 m: control = stored e006u 150/150; switch off = e006u 150/150; = land1012 lists {'v69a': 150, 'v69b': 150, 'v69c': 150}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check mx c2 m: control = stored e006u 380/380; switch off = e006u 380/380; = land1012 lists {'v69a': 380, 'v69b': 380, 'v69c': 380}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check mx dA m: control = stored e006u 325/325; switch off = e006u 325/325; = land1012 lists {'v69a': 325, 'v69b': 325, 'v69c': 325}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check mx dB m: control = stored e006u 325/325; switch off = e006u 325/325; = land1012 lists {'v69a': 325, 'v69b': 325, 'v69c': 325}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check mx dA u: control = stored e006u 325/325; switch off = e006u 325/325; = land1012 lists {'v69a': 325, 'v69b': 325, 'v69c': 325}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check mx dB u: control = stored e006u 325/325; switch off = e006u 325/325; = land1012 lists {'v69a': 325, 'v69b': 325, 'v69c': 325}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True

### mx v69a
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 150 | 0.8337 | 0.8356 | +0.0019 ± 0.0036 | 1 / 2 | 0 / 0 | 19 (1) | 1 / 0 | 25, 0, 0 |
| c2 m | 380 | 0.7409 | 0.7360 | -0.0048 ± 0.0043 | 5 / 7 | 0 / 0 | 159 (13) | 2 / 4 | 192, 0, 0 |
| dA m | 325 | 0.3967 | 0.4135 | +0.0168 ± 0.0068 | 25 / 13 | 0 / 0 | 128 (34) | 11 / 3 | 158, 0, 0 |
| dB m | 325 | 0.1454 | 0.1638 | +0.0184 ± 0.0055 | 25 / 10 | 0 / 0 | 126 (36) | 6 / 1 | 152, 0, 0 |
| dA u | 325 | 0.5008 | 0.5100 | +0.0092 ± 0.0057 | 16 / 13 | 0 / 0 | 127 (24) | 8 / 2 | 157, 0, 0 |
| dB u | 325 | 0.2498 | 0.2663 | +0.0165 ± 0.0065 | 24 / 11 | 0 / 0 | 125 (36) | 10 / 2 | 152, 0, 0 |
| **composite, arm m** | 380 | | | **+0.0072 ± 0.0033** | 42 / 21 | | | | |
| **composite, arm u** | 380 | | | **+0.0049 ± 0.0032** | 36 / 20 | | | | |

### mx v69b
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 150 | 0.8337 | 0.8370 | +0.0033 ± 0.0033 | 1 / 0 | 0 / 0 | 2 (2) | 1 / 0 | 0, 2, 0 |
| c2 m | 380 | 0.7409 | 0.7457 | +0.0048 ± 0.0032 | 5 / 2 | 0 / 0 | 8 (8) | 5 / 1 | 0, 8, 0 |
| dA m | 325 | 0.3967 | 0.3931 | -0.0036 ± 0.0022 | 0 / 3 | 0 / 0 | 4 (4) | 0 / 2 | 0, 4, 0 |
| dB m | 325 | 0.1454 | 0.1498 | +0.0044 ± 0.0031 | 2 / 0 | 0 / 0 | 4 (4) | 2 / 0 | 0, 4, 0 |
| dA u | 325 | 0.5008 | 0.4972 | -0.0036 ± 0.0022 | 0 / 3 | 0 / 0 | 4 (4) | 0 / 2 | 0, 4, 0 |
| dB u | 325 | 0.2498 | 0.2541 | +0.0044 ± 0.0031 | 2 / 0 | 0 / 0 | 4 (4) | 2 / 0 | 0, 4, 0 |
| **composite, arm m** | 380 | | | **+0.0021 ± 0.0015** | 8 / 4 | | | | |
| **composite, arm u** | 380 | | | **+0.0021 ± 0.0015** | 8 / 4 | | | | |

### mx v69c
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 150 | 0.8337 | 0.8336 | -0.0001 ± 0.0001 | 0 / 1 | 0 / 0 | 57 (0) | 0 / 0 | 0, 0, 0 |
| c2 m | 380 | 0.7409 | 0.7411 | +0.0002 ± 0.0002 | 2 / 3 | 2 / 0 | 163 (0) | 0 / 0 | 0, 0, 0 |
| dA m | 325 | 0.3967 | 0.3950 | -0.0017 ± 0.0016 | 1 / 7 | 1 / 0 | 144 (2) | 0 / 1 | 0, 0, 0 |
| dB m | 325 | 0.1454 | 0.1466 | +0.0012 ± 0.0018 | 5 / 3 | 0 / 0 | 147 (1) | 1 / 0 | 0, 0, 0 |
| dA u | 325 | 0.5008 | 0.5005 | -0.0003 ± 0.0003 | 1 / 6 | 1 / 1 | 144 (0) | 0 / 0 | 0, 0, 0 |
| dB u | 325 | 0.2498 | 0.2491 | -0.0006 ± 0.0009 | 2 / 4 | 0 / 1 | 150 (0) | 0 / 0 | 0, 0, 0 |
| **composite, arm m** | 380 | | | **-0.0000 ± 0.0005** | 7 / 10 | | | | |
| **composite, arm u** | 380 | | | **-0.0002 ± 0.0003** | 4 / 9 | | | | |

### mx v69d
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 150 | 0.8337 | 0.8337 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| c2 m | 380 | 0.7409 | 0.7409 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dA m | 325 | 0.3967 | 0.3967 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dB m | 325 | 0.1454 | 0.1454 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dA u | 325 | 0.5008 | 0.5023 | +0.0015 ± 0.0015 | 1 / 0 | 0 / 0 | 1 (1) | 1 / 0 | 0, 0, 1 |
| dB u | 325 | 0.2498 | 0.2644 | +0.0146 ± 0.0052 | 8 / 0 | 0 / 0 | 8 (8) | 8 / 0 | 0, 0, 8 |
| **composite, arm m** | 380 | | | **+0.0000 ± 0.0000** | 0 / 0 | | | | |
| **composite, arm u** | 380 | | | **+0.0040 ± 0.0014** | 8 / 0 | | | | |

### mx v69e
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 150 | 0.8337 | 0.8389 | +0.0052 ± 0.0049 | 2 / 2 | 0 / 0 | 21 (3) | 2 / 0 | 25, 2, 0 |
| c2 m | 380 | 0.7409 | 0.7409 | +0.0000 ± 0.0054 | 10 / 9 | 0 / 0 | 167 (21) | 7 / 5 | 192, 8, 0 |
| dA m | 325 | 0.3967 | 0.4099 | +0.0132 ± 0.0072 | 25 / 16 | 0 / 0 | 132 (38) | 11 / 5 | 158, 4, 0 |
| dB m | 325 | 0.1454 | 0.1682 | +0.0228 ± 0.0063 | 27 / 10 | 0 / 0 | 130 (40) | 8 / 1 | 152, 4, 0 |
| dA u | 325 | 0.5008 | 0.5064 | +0.0056 ± 0.0061 | 16 / 16 | 0 / 0 | 131 (28) | 8 / 4 | 157, 4, 0 |
| dB u | 325 | 0.2498 | 0.2706 | +0.0209 ± 0.0071 | 26 / 11 | 0 / 0 | 129 (40) | 12 / 2 | 152, 4, 0 |
| **composite, arm m** | 380 | | | **+0.0093 ± 0.0036** | 50 / 25 | | | | |
| **composite, arm u** | 380 | | | **+0.0069 ± 0.0035** | 44 / 24 | | | | |
check p2 c1 m: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check p2 c2 m: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check p2 dA m: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check p2 dB m: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check p2 dA u: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True
check p2 dB u: control = stored e006u 226/226; switch off = e006u 226/226; = land1012 lists {'v69a': 226, 'v69b': 226, 'v69c': 226}; lists valid (1-25 rows, unique metric keys) True; self-test ok and 0 fallbacks True

### p2 v69a
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 226 | 0.9371 | 0.9349 | -0.0022 ± 0.0022 | 0 / 1 | 0 / 0 | 57 (1) | 0 / 1 | 68, 0, 0 |
| c2 m | 226 | 0.8232 | 0.8156 | -0.0076 ± 0.0046 | 2 / 5 | 0 / 0 | 75 (6) | 0 / 3 | 99, 0, 0 |
| dA m | 226 | 0.5098 | 0.5293 | +0.0194 ± 0.0069 | 16 / 4 | 0 / 0 | 75 (10) | 5 / 0 | 99, 0, 0 |
| dB m | 226 | 0.2164 | 0.2286 | +0.0122 ± 0.0048 | 13 / 3 | 0 / 0 | 79 (15) | 3 / 0 | 101, 0, 0 |
| dA u | 226 | 0.6938 | 0.6966 | +0.0028 ± 0.0055 | 7 / 4 | 0 / 0 | 76 (6) | 2 / 1 | 100, 0, 0 |
| dB u | 226 | 0.4366 | 0.4434 | +0.0067 ± 0.0077 | 13 / 7 | 0 / 0 | 79 (12) | 6 / 2 | 102, 0, 0 |
| **composite, arm m** | 226 | | | **+0.0054 ± 0.0027** | 28 / 7 | | | | |
| **composite, arm u** | 226 | | | **-0.0001 ± 0.0034** | 18 / 9 | | | | |

### p2 v69b
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 226 | 0.9371 | 0.9371 | +0.0000 ± 0.0031 | 1 / 1 | 0 / 0 | 3 (3) | 1 / 1 | 0, 3, 0 |
| c2 m | 226 | 0.8232 | 0.8232 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 2 (2) | 0 / 0 | 0, 2, 0 |
| dA m | 226 | 0.5098 | 0.5087 | -0.0011 ± 0.0008 | 0 / 2 | 0 / 0 | 2 (2) | 0 / 0 | 0, 2, 0 |
| dB m | 226 | 0.2164 | 0.2197 | +0.0033 ± 0.0033 | 1 / 0 | 0 / 0 | 3 (3) | 1 / 0 | 0, 3, 0 |
| dA u | 226 | 0.6938 | 0.6927 | -0.0011 ± 0.0008 | 0 / 2 | 0 / 0 | 2 (2) | 0 / 0 | 0, 2, 0 |
| dB u | 226 | 0.4366 | 0.4399 | +0.0033 ± 0.0033 | 1 / 0 | 0 / 0 | 3 (3) | 1 / 0 | 0, 3, 0 |
| **composite, arm m** | 226 | | | **+0.0006 ± 0.0012** | 2 / 3 | | | | |
| **composite, arm u** | 226 | | | **+0.0006 ± 0.0012** | 2 / 3 | | | | |

### p2 v69c
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 226 | 0.9371 | 0.9371 | -0.0000 ± 0.0000 | 0 / 1 | 0 / 0 | 118 (0) | 0 / 0 | 0, 0, 0 |
| c2 m | 226 | 0.8232 | 0.8228 | -0.0005 ± 0.0008 | 5 / 3 | 1 / 1 | 118 (0) | 0 / 0 | 0, 0, 0 |
| dA m | 226 | 0.5098 | 0.5107 | +0.0009 ± 0.0012 | 7 / 6 | 6 / 0 | 116 (0) | 0 / 0 | 0, 0, 0 |
| dB m | 226 | 0.2164 | 0.2169 | +0.0005 ± 0.0032 | 4 / 5 | 0 / 0 | 117 (3) | 1 / 1 | 0, 0, 0 |
| dA u | 226 | 0.6938 | 0.6962 | +0.0024 ± 0.0022 | 3 / 3 | 0 / 0 | 118 (1) | 1 / 0 | 0, 0, 0 |
| dB u | 226 | 0.4366 | 0.4367 | +0.0001 ± 0.0032 | 4 / 3 | 0 / 0 | 118 (2) | 1 / 1 | 0, 0, 0 |
| **composite, arm m** | 226 | | | **+0.0002 ± 0.0009** | 14 / 14 | | | | |
| **composite, arm u** | 226 | | | **+0.0005 ± 0.0010** | 8 / 9 | | | | |

### p2 v69d
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 226 | 0.9371 | 0.9371 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| c2 m | 226 | 0.8232 | 0.8232 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dA m | 226 | 0.5098 | 0.5098 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dB m | 226 | 0.2164 | 0.2164 | +0.0000 ± 0.0000 | 0 / 0 | 0 / 0 | 0 (0) | 0 / 0 | 0, 0, 0 |
| dA u | 226 | 0.6938 | 0.7026 | +0.0088 ± 0.0044 | 4 / 0 | 0 / 0 | 4 (4) | 4 / 0 | 0, 0, 4 |
| dB u | 226 | 0.4366 | 0.5399 | +0.1032 ± 0.0144 | 43 / 0 | 0 / 0 | 43 (43) | 43 / 0 | 0, 0, 43 |
| **composite, arm m** | 226 | | | **+0.0000 ± 0.0000** | 0 / 0 | | | | |
| **composite, arm u** | 226 | | | **+0.0280 ± 0.0040** | 43 / 0 | | | | |

### p2 v69e
| type, arm | n | e006u | patch | difference ± SE | wins / losses | in / out | lists (first rows) that differ | first rows: answer gained / lost rank 1 | rule fired (joint, shield, promoted-first) |
|---|---|---|---|---|---|---|---|---|---|
| c1 m | 226 | 0.9371 | 0.9349 | -0.0022 ± 0.0038 | 1 / 2 | 0 / 0 | 60 (4) | 1 / 2 | 68, 3, 0 |
| c2 m | 226 | 0.8232 | 0.8156 | -0.0076 ± 0.0046 | 2 / 5 | 0 / 0 | 77 (8) | 0 / 3 | 99, 2, 0 |
| dA m | 226 | 0.5098 | 0.5282 | +0.0183 ± 0.0069 | 16 / 6 | 0 / 0 | 77 (12) | 5 / 0 | 99, 2, 0 |
| dB m | 226 | 0.2164 | 0.2319 | +0.0155 ± 0.0058 | 14 / 3 | 0 / 0 | 82 (18) | 4 / 0 | 101, 3, 0 |
| dA u | 226 | 0.6938 | 0.6955 | +0.0017 ± 0.0056 | 7 / 6 | 0 / 0 | 78 (8) | 2 / 1 | 100, 2, 0 |
| dB u | 226 | 0.4366 | 0.4467 | +0.0100 ± 0.0084 | 14 / 7 | 0 / 0 | 82 (15) | 7 / 2 | 102, 3, 0 |
| **composite, arm m** | 226 | | | **+0.0060 ± 0.0030** | 30 / 10 | | | | |
| **composite, arm u** | 226 | | | **+0.0005 ± 0.0036** | 20 / 12 | | | | |

ALL_CHECKS_OK
