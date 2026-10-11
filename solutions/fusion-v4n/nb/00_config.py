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
                                  #   'fs' = below RANKER_LIB_GATE a second engine pass with the fold-safe FPNet (engine bank
                                  #   and model_views) and the ranker fitted to its features (casmi26-ours-ranker-fs) orders
                                  #   the list; at or above the gate the first pass stays as it is | 'fs_avg' = below the gate
                                  #   the two chains' ranks are averaged (nb/embed/fs_pass.py, reviews/fsranker-2026-10-06.md).
                                  #   Both need RANKER_LIB_GATE; one extra engine pass per molecule below the gate
    RANKER_LIB_GATE=None,         # lib_max >= this keeps the public ranker whatever RANKER says; None = no gate
    RANKER_FS_W=None,             # 'fs_avg': weight of the fold-safe chain in the rank average; None = 0.5 (v33b)
    RANKER_FS_FADE=None,          # 'fs_avg': [lo, hi] = that weight is full up to lib_max lo and falls linearly to 0 at hi (set
                                  #   RANKER_LIB_GATE = hi); None = the same weight everywhere below the gate (v33b)
    RANKER_FS_DS=None,            # dataset folder of the fitted ranker; None = 'casmi26-ours-ranker-fs' (v33b). 'casmi26-ours-ranker-fs2'
                                  #   = the refit of reviews/fsranker2-2026-10-07.md
    RANKER_FS_NET=None,           # glob (below /kaggle/input) of the fold-safe FPNet of the second pass; None = the public
                                  #   'casmi26-fold-safe-fpnet/**/fpnet_fold0.pt' (v33b / v59a). 'casmi26-ours-fpnet-fs/**/fpnet_own_fs.pt' = our own
                                  #   network (clean/train_fpnet.py, reviews/ownfp-2026-10-10.md); set RANKER_FS_DS to the ranker fitted on it
    ADDUCT_TIE=None,              # the public engine labels a polarity's merged spectrum with its most frequent adduct and breaks a
                                  #   tie by set order = the kernel's hash seed, so the base lists of tied molecules differ between
                                  #   sessions (reviews/determ-2026-10-07.md). 'train' = upstream's own rule of v1engine.merged_adduct
                                  #   (most frequent in the training spectra) | 'first' = first in the molecule's spectra | 'mean' =
                                  #   logits averaged over the tied adducts (nb/embed/determ.py). None = set order, as before
    HASHSEED=None,                # int: PYTHONHASHSEED exported before any subprocess starts (engine 2, PubChem channel, ICEBERG,
                                  #   GLACIER). It cannot change the running kernel's own seed: that is what ADDUCT_TIE is for
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
    ENG_REPS=1,                   # engine-2 analog search: representative spectra per library structure (1 = the reference's one;
                                  #   k > 1 = the first k by its own order, best similarity per structure; reviews/keyvote-2026-10-09.md)
    ENG_EXTRA_FPNET=None,         # 'owner/slug' of a dataset with an FPNet of prvsiyan's architecture (fp_*.pt, 6,930 bits)
                                  #   that is averaged into engine-2's fingerprint logits (a file without 'merged' in its
                                  #   name joins the per-spectrum group). 'goooman/casmi26-fp-npx4-soup' = the teammate's
                                  #   np1 (0.413 on v4b 0.409); reviews/np-eng2-2026-10-05.md. 'np-ours' = our own
                                  #   fine-tune of the public fp_single_s2 (dataset <user>/casmi26-fp-np-ours;
                                  #   train/train_eng_fpnet.py, reviews/np-ours-2026-10-05.md)
    FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FUSE_ENG_K=40,   # FUSE_ENG_K: engine-2 depth used in the fusion
    # --- PubChem popularity prior on pool candidates (bobthebot369 v10, LB 0.401 used 0.15); 0 = off
    POP_MU=0.0,
    POP_SPLIT=False,              # the prior's substance count without the records deposited by spectral libraries (MassBank Europe,
                                  #   MoNA, NIST MSDC, RIKEN NPDepo), for the pool rows listed in <user>/casmi26-pop-split; every other
                                  #   row keeps today's value (nb/embed/pop_split.py, reviews/popsplit-2026-10-08.md)
    # --- MetFrag-parsimony re-score (nb/embed/frag_rescore.py); 0 = off. 'gap' = only molecules ICE cannot score
    FRAG_LAM=0.0, FRAG_MODE='all',
    # --- more fusion knobs (fusion_core.DEFAULTS); None = same as the base weight / off
    POST_ICE_LAM=None, POST_GL_LAM=None,   # stage-C (post-fusion) re-rank weights
    ICE_LAM_LIB=None, GL_LAM_LIB=None,     # weights for library hits (lib_max >= LIB_TAU)
    FUSE_SKIP_LIB=None,                    # skip the engine-2 fusion when lib_max >= this
    OUT_TOPK=None,                         # diagnostic: write only the first k rows per molecule (leaderboard probe of rank-1 / rank-2 mass)
    FILL_25=False,                         # top up lists shorter than 25 from the other channels
    KEY_NOSTEREO=False,                    # final list de-duplicated by the metric key of 2026-10-07 (stereo stripped before the tautomer step) before the cut to 25 (reviews/emass-2026-10-08.md)
    FUSE_KEY_VOTE=None,                    # engine fusion keyed by the stereo-free metric key, one vote per key and engine: 'best' (first
                                           #   rank of the key in each list) | 'compact' (ranks counted over distinct keys, flexonafft v80);
                                           #   None = off (reviews/keyvote-2026-10-09.md)
    POP_LIB_OFF=False, FRAG_LIB_OFF=False, # no popularity / frag term for library hits
    GATE_TAU=None,                         # lib_max threshold of the *_LIB gates; None = LIB_TAU
    FUSE_MIN_LIB=None,                     # fuse engine-2 only when lib_max >= this (it needs library analogs)
    FUSE_INJ_CREDIT=0.0,                   # engine fusion: joined PubChem rows get this share of the engine-2 weight at
                                           #   their own rank (engine-2 cannot list them); 1.0 = as if both engines agreed
    JOIN_FWD_A=None,                       # stage A: ICEBERG / GLACIER terms of joined PubChem rows times this (0.25);
                                           #   None = off (reviews/calib-2026-10-05.md)
    FWD_A_SCALE=None,                      # stage A, base list only: [ice, gl] factors on ICE_LAM / GL_LAM (e.g. [0.33, 0.33]); the
                                           #   PubChem list and stage C keep the full weights; None = off (reviews/fsranker2-2026-10-07.md)
    POP_GROUP_MU=None,                     # stage A: in-group popularity term, mu x z(popularity) inside each same-formula group of
                                           #   pool rows (POP_MU keeps acting across the list; 0 = in-group only); None = off
    POP_GROUP_MAX_LIB=None,                #   ... only for molecules with lib_max below this (reviews/popgroup-2026-10-08.md)
    NP_MU=0.0, NP_JOIN_MU=0.0,             # natural-product-likeness term on pool rows / joined PubChem rows (fusion_core.np_reorder).
                                           #   Replayed locally, lost; NO scorer is shipped with the notebook, so cell 08 refuses
                                           #   non-zero values (reviews/nplike-2026-10-08.md)
    ADD_JOIN_MU=0.0,                       # adduct-evidence term on joined PubChem rows: Bernoulli log-likelihood of the molecule's
                                           #   observed adduct set under seven LightGBM models (nb/embed/adduct_prior.py, dataset
                                           #   casmi26-ours-adduct-prior), standardised over the list; 0 = off (reviews/newinfo-2026-10-08.md)
    PC_FWD_T1=None, PC_FWD_N=25,           # PubChem proposals get the early slots when the best one beats every
                                           #   same-formula base candidate in ICEBERG and GLACIER by more than this (0.0)
    JOIN_POP_MU=0.0,                       # popularity term on joined PubChem rows: f = z + JOIN_POP_MU x (log1p substance count + log1p PubMed
                                           #   count) of the row's PubChem-tier record (casmi26-pubchem-popularity-prior, the table POP_MU reads
                                           #   for pool rows; nb/embed/pc_pop_side.py); 0 = off (reviews/graft-2026-10-10.md)
    JOIN_POP_FREE=False,                   # ... False: joined rows re-sorted among their own slots; True: together with the pool rows
    PC_PROMOTE=None,                       # 'e006' = the public line's promotion: a second PubChem-channel pass with the popularity prior
                                           #   (0.25 / 200, about 1 h on the hidden set); its first row after the forward re-rank leads the list
                                           #   when lib_max < LIB_TAU and ((S > PROM_S_TAU and popularity >= PROM_POP_TAU) or rel > PROM_REL) |
                                           #   'strict' = PROM_MODEL decides | 'observe' = the pass runs, nothing moves. None = off
    PROM_S_TAU=6.0, PROM_POP_TAU=5.0, PROM_REL=250.0,
    PROM_STAGE='B',                        # 'B' = before the engine fusion (as in e006) | 'C' = after it (the row is the final first row)
    PROM_MODEL=None,                       # 'strict': dict(feat, mean, sd, coef, icpt, thr) of a logistic model (reviews/graft-2026-10-10.md)
    FWD_GL_ALONE=False,                    # robustness: a molecule without an ICEBERG score but with GLACIER scores (ICEBERG crashed or
                                           #   hit ICE_BUDGET) is re-ranked by GLACIER alone instead of losing both models and falling to
                                           #   the fragment 'gap' term; changes nothing while ICEBERG is healthy
                                           #   (docs/landscape/canqiang-e006-2026-10-10.md)
    # --- analog generator channel (nb/embed/gen_analog.py, reviews/gen3-2026-10-10.md); None = off
    GEN_ANALOG=None,              # 'slots' = up to GEN_ANALOG_K new structures (one rule step from the best fingerprint-ranked pool
                                  #   parents, formulas of the base list) take the positions from GEN_ANALOG_POS of the final list |
                                  #   'gate' = they compete by a fitted probability, only below lib_max 0.7 and below a top score of the
                                  #   fold-safe ranker (needs RANKER 'fs' / 'fs_avg'). After the fusion; one CPU core, about 1 s per molecule
    GEN_ANALOG_POS=6, GEN_ANALOG_K=5,
    # --- second pipeline (nb/09_e006.py, nb/embed/e006_stage.py, reviews/e006-2026-10-10.md); None = off = nothing runs
    E006_FUSE=None,               # 'rrf' | 'fill' | 'agree': after our own submission.csv is written, the public notebook "E006 Window 40
                                  #   ICE Fallback True" (canqiang, Apache-2.0, 0.440; reference/e006/) runs as a script in its own folder and
                                  #   its table is fused with ours on the metric key. 'rrf' = 1/(k + rank) summed over the two lists;
                                  #   'fill' = ours first, their rows that are not in ours inserted from E006_FILL_POS; 'agree' = rows both
                                  #   hold in their first five lead, the rest as 'rrf'. A failed or late stage keeps our own table
    E006_RRF_K=3.0, E006_FILL_POS=6,
    E006_RRF_W=1.0,               # 'rrf': weight of the second list, w/(k + rank); 1 = equal weights (the pre-registered rule, v65a)
    E006_SHARE_ENG=True,          # hand the second pipeline our engine-2 lists (same embedded code and arguments) instead of a second run;
                                  #   only when the ENG_* knobs are the reference's, otherwise it runs its own engine 2
    E006_DEADLINE_S=27600,        # the second pipeline is stopped when the notebook has run this long (7 h 40 min); our table then stays
    # --- "e006 improved" (nb/e006imp_stage.py, nb/e006imp_final.py, reviews/e006imp-2026-10-10.md); None = off = the two cells are not in the notebook
    E006_BASE=None,               # 'ours' = the public notebook E006 (canqiang, Apache-2.0, 0.440) is the pipeline: after our base lists (cell 06)
                                  #   its script runs with OUR base list in place of its own (join + rank average below the gates), everything
                                  #   else by its own cells (its PubChem channel with popularity, slots, promotion, fusion, tail). Its table is
                                  #   the submission; cells 07 / 08 then skip the forward models. A failed or late stage -> cells 07 / 08 run in
                                  #   full and our own table (this preset without the option) is the submission
    E006_BASE_DEADLINE_S=22200,   # the stage is stopped when the notebook has run this long (6 h 10 min): our own forward stage still fits
    E006_PROM_REL=None,           # its promotion's `rel > 250` branch with another threshold (1e12 = branch off); None = its own 250
    E006_GL_ALL=False,            # its GLACIER scores every covered spectrum instead of [M+H]+ only
    E006_PROM_FINAL=False,        # a row promoted by its popularity branch leads the final list too (after its fusion, key vote, tail)
    E006_JOIN_POP_MU=0.0,         # joined rows of the handed-over base list re-sorted among their own slots by z + this x popularity; 0 = off
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
