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
    # --- library gate + PubChem merge
    LIB_TAU=0.9,                  # lib_max >= LIB_TAU -> library hit, list left untouched
    REL_TH=600.0,                 # PubChem top f.z minus best pool f.z above this -> aggressive slots
    SLOTS_AGG=[2, 4, 6, 8, 10],
    SLOTS_GENTLE=[4, 8, 12, 16, 20],
    # --- PubChem-only channel
    USE_PC=True, PC_N1=5000, PC_WORKERS=4,
    PC_POP_LAM=0.0, PC_POP_UNION=200,   # popularity prior inside the PubChem channel (wangpenghua v29 used 0.25 / 200); 0 = off
    # --- ICEBERG / GLACIER re-scoring
    USE_ICE=True, ICE_LAM=1.0, ICE_BUDGET=5400, ICE_PC=True,
    ICE_UNION=True, ICE_UNION_K=40,   # also score engine-2 candidates
    USE_GL=True, GL_LAM=1.0, GL_BUDGET=4000,
    GL_MH_ONLY=False,             # GLACIER scores [M+H]+ spectra only (wangpenghua v29)
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
    STAGE_TIMEOUT_S=4 * 3600,
)
