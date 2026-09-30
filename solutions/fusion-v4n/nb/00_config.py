# casmi26-ours — every tunable knob lives here. Defaults reproduce the reference notebook
# "CASMI26 | v4n + Engine Fusion + Union [LB 0.399]" (seyitkaangunes, Apache-2.0); change one knob per experiment
# and log it in EXPERIMENTS.md.
CFG = dict(
    VERSION='ours-v1-baseline',
    # --- v1 engine + ranker (base lists)
    TOPN=60,                      # base list depth handed to re-scoring
    FP_BANK='full1',              # 'full1' = casmi26-fpnet-full1 (v4m); 'A' = fpnet_0.pt of casmi26-v4b-models
    # --- library gate + PubChem merge
    LIB_TAU=0.9,                  # lib_max >= LIB_TAU -> library hit, list left untouched
    REL_TH=600.0,                 # PubChem top f.z minus best pool f.z above this -> aggressive slots
    SLOTS_AGG=[2, 4, 6, 8, 10],
    SLOTS_GENTLE=[4, 8, 12, 16, 20],
    # --- PubChem-only channel
    USE_PC=True, PC_N1=5000, PC_WORKERS=4,
    # --- ICEBERG / GLACIER re-scoring
    USE_ICE=True, ICE_LAM=1.0, ICE_BUDGET=5400, ICE_PC=True,
    ICE_UNION=True, ICE_UNION_K=40,   # also score engine-2 candidates
    USE_GL=True, GL_LAM=1.0, GL_BUDGET=4000,
    # --- engine-2 (prvsiyan two-ranker, BIO + AFIX) and its fusion
    USE_ENG=True, ENG_W_PV=0.88, ENG_W_A=[0.35, 0.55], ENG_SEEDS=[0, 1], ENG_N_ANALOG=200, ENG_ORDER_K=80, ENG_TOPK=40,
    FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40,
    # --- run control
    SMOKE_N=12,                   # commit run on the visible test: only this many molecules ...
    SMOKE_ICE_BUDGET=300,
    FULL_ON_COMMIT=False,         # ... unless True (full visible test, for stage dumps + local re-fusion)
    DUMP_STAGES=True,             # write /kaggle/working/stage_cache/*.json
    STAGE_TIMEOUT_S=4 * 3600,
)
