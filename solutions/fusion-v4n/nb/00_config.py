# casmi26-ours — every tunable knob lives here. Defaults reproduce the reference notebook
# "CASMI26 | v4n + Engine Fusion + Union [LB 0.399]" (seyitkaangunes, Apache-2.0); change one knob per experiment
# and log it in EXPERIMENTS.md.
CFG = dict(
    VERSION='ours-v1-baseline',
    # --- v1 engine + ranker (base lists)
    TOPN=60,                      # base list depth handed to re-scoring
    FP_BANK='full1',              # 'full1' = casmi26-fpnet-full1 (v4m, trained on everything); 'A' = fpnet_0.pt of
                                  #   casmi26-v4b-models (never saw enveda-np-examples); 'ens' = both; 'fold0' = the
                                  #   fold-safe checkpoint of dmitriigluzdov (validation only)
    RANKER='public',              # base-list score: 'public' (the public ranker as it is) | 'public_free' (public ranker,
                                  #   rank features with order-free ties) | 'ours' (casmi26-ours-ranker) | 'blend'
                                  #   (z(ours) + z(public_free)); see reviews/ranker-2026-10-02.md
    RANKER_LIB_GATE=None,         # lib_max >= this keeps the public ranker whatever RANKER says; None = no gate
    # --- library gate + PubChem merge
    LIB_TAU=0.9,                  # lib_max >= LIB_TAU -> library hit, list left untouched
    REL_TH=600.0,                 # PubChem top f.z minus best pool f.z above this -> aggressive slots
    SLOTS_AGG=[2, 4, 6, 8, 10],
    SLOTS_GENTLE=[4, 8, 12, 16, 20],
    # --- PubChem-only channel
    USE_PC=True, PC_N1=5000, PC_WORKERS=4,
    PC_POP_LAM=0.0, PC_POP_UNION=200,   # popularity prior inside the PubChem channel (wangpenghua v29 used 0.25 / 200); 0 = off
    PC_MODELS=None,               # 'hn' = the candidates of the join (PC_JOIN_N) are selected by the FPNet fine-tuned
                                  #   against PubChem-window negatives (casmi26-fpnet-pc-hn, reviews/hardneg-2026-10-05.md)
                                  #   in a second channel pass; the fusion's PubChem list keeps public A+B. None = A+B
    PC_PPM=10.0,                  # mass window of the PubChem-only channel in ppm (5.0 with the offset below:
                                  #   reviews/masscal-2026-10-04.md)
    MASS_OFFSET_BY_ADDUCT=None,   # 'enveda' = per-adduct ppm offset of the Enveda timsTOF libraries removed from the
                                  #   channel's target mass (nb/embed/pc/mass_window.py); None = off
    PC_JOIN_N=0,                  # best N PubChem-only candidates enter the v1 engine and are placed by the ranker
                                  #   (nb/embed/pc_join.py, reviews/pcjoin-2026-10-02.md); 0 = off. Needs PC_POP_LAM = 0
    PC_JOIN_ORDER='first',        # 'first': the generator treats injected keys as pool keys | 'after'
    PC_JOIN_MAX_LIB=None,         # no injection when the engine's lib_max >= this (0.7 = class-1-like lists as today)
    PC_JOIN_MIN_REL=None,         # only proposals with f.z - best pool f.z above this (not recommended)
    PC_JOIN_CONF=None,            # join only when conf_gate.p_present of the list WITHOUT injection is below this (0.5 / 0.7;
                                  #   reviews/confgate-2026-10-04.md); on top of PC_JOIN_MAX_LIB. None = off. One extra engine
                                  #   pass per flagged molecule
    PC_JOIN_CONF_DEEP=None, PC_JOIN_N_DEEP=0,   # lists with p_present below PC_JOIN_CONF_DEEP get PC_JOIN_N_DEEP candidates
    # --- ICEBERG / GLACIER re-scoring
    USE_ICE=True, ICE_LAM=1.0, ICE_BUDGET=5400, ICE_PC=True,
    ICE_FT=False,                 # ICEBERG with the intensity model fine-tuned on the competition's training spectra
                                  #   (ahmedberatozer/casmi26-ice-ft, inten.pt; same architecture as the stock checkpoint)
    ICE_UNION=True, ICE_UNION_K=40,   # also score engine-2 candidates
    USE_GL=True, GL_LAM=1.0, GL_BUDGET=4000,
    GL_MH_ONLY=False,             # GLACIER scores [M+H]+ spectra only (wangpenghua v29)
    ICE_SURROGATE=False,          # score [M+NH4]+ / water-loss spectra against the [M+H]+ prediction (nb/embed/surrogate.py):
                                  #   'gap' = only molecules without an [M+H]+ / [M+Na]+ spectrum, 'all' = every molecule
    GL_SURROGATE=False,           # hand the surrogate spectra to GLACIER as well (untested)
    # --- engine-2 (prvsiyan two-ranker, BIO + AFIX) and its fusion
    USE_ENG=True, ENG_W_PV=0.88, ENG_W_A=[0.35, 0.55], ENG_SEEDS=[0, 1], ENG_N_ANALOG=200, ENG_ORDER_K=80, ENG_TOPK=40,
    FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FUSE_ENG_K=40,   # FUSE_ENG_K: engine-2 depth used in the fusion
    # --- PubChem popularity prior on pool candidates (bobthebot369 v10, LB 0.401 used 0.15); 0 = off
    POP_MU=0.0,
    # --- MetFrag-parsimony re-score (nb/embed/frag_rescore.py); 0 = off. 'gap' = only molecules ICE cannot score
    FRAG_LAM=0.0, FRAG_MODE='all',
    # --- more fusion knobs (fusion_core.DEFAULTS); None = same as the base weight / off
    POST_ICE_LAM=None, POST_GL_LAM=None,   # stage-C (post-fusion) re-rank weights
    ICE_LAM_LIB=None, GL_LAM_LIB=None,     # weights for library hits (lib_max >= LIB_TAU)
    FUSE_SKIP_LIB=None,                    # skip the engine-2 fusion when lib_max >= this
    FILL_25=False,                         # top up lists shorter than 25 from the other channels
    POP_LIB_OFF=False, FRAG_LIB_OFF=False, # no popularity / frag term for library hits
    GATE_TAU=None,                         # lib_max threshold of the *_LIB gates; None = LIB_TAU
    FUSE_MIN_LIB=None,                     # fuse engine-2 only when lib_max >= this (it needs library analogs)
    FUSE_INJ_CREDIT=0.0,                   # engine fusion: joined PubChem rows get this share of the engine-2 weight at
                                           #   their own rank (engine-2 cannot list them); 1.0 = as if both engines agreed
    PC_FWD_T1=None, PC_FWD_N=25,           # PubChem proposals get the early slots when the best one beats every
                                           #   same-formula base candidate in ICEBERG and GLACIER by more than this (0.0)
    # --- validation mode (commit runs only; the scoring rerun ignores it): hold out VAL_LIB structures, drop ALL
    #     their spectra from the library, score them as the test set -> stage_cache/val_labels.csv
    VALIDATION=False, VAL_LIB='enveda-np-examples', VAL_N=250, VAL_SEED=0, VAL_MAX_SPEC=16,
    VAL_PURGE='all',              # 'all' = class-2-like (no spectra left); 'lib' = class-1-like (other libraries stay)
    VAL_SET='np_examples',        # 'np_examples' = VAL_LIB structures (famous NPs); 'fold0_np' = NP-catalogue structures
                                  #   unseen by the fold-safe FPNet (ordinary NPs; use FP_BANK='fold0', VAL_MAX_SPEC ~6)
    VAL_EXTRAS=False,             # validation runs also dump pc_pop.json (PubChem channel with popularity) and gl_mh.json
    # --- run control
    SMOKE_N=12,                   # commit run on the visible test: only this many molecules ...
    SMOKE_ICE_BUDGET=300,
    FULL_ON_COMMIT=False,         # ... unless True (full visible test, for stage dumps + local re-fusion)
    DUMP_STAGES=True,             # write /kaggle/working/stage_cache/*.json
    STRICT=False,                 # raise at the end if any stage fell back (engine-2, PubChem channel, join, ranker,
                                  #   ICEBERG, GLACIER, frag): a diagnostic submission then fails instead of scoring low
    STAGE_TIMEOUT_S=4 * 3600,
)
