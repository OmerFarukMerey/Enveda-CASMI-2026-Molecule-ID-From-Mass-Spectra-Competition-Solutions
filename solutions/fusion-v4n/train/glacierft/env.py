"""GLACIER fine-tune environment: import the official ms-pred code (coleygroup/ms-pred @ 708148c2a8eb) without installing anything.

sys.path order: our pygmtools stand-in (shim/), the GLACIER package's pure-torch stand-ins for dgl / torch_scatter / LinSATNet
(casmi26-glacier shim/, the same ones stack1's runner uses), the official ms-pred checkout's src/, and the wheel site of the
casmi26-iceberg dataset (pytorch_lightning / torchmetrics / einops). Modules ms_pred imports at module level but this fine-tune never
uses (h5py, multiprocess, ray-based tuning, plotting) are served as inert stubs, as in the package runner gl_runner.py.

Paths are overridable: $MSPRED_SRC, $GLACIER_PKG, $ICE_SITE.
"""
import importlib, os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
MSPRED_SRC = os.environ.get('MSPRED_SRC', r'C:\Users\hemre\casmi-work\ext\ms-pred\src')
GLACIER_PKG = os.environ.get('GLACIER_PKG', r'D:\casmi-data\coverage_study\glacier')
ICE_SITE = os.environ.get('ICE_SITE', r'D:\casmi-data\coverage_study\ice_site_win')
BASE_CKPT = os.path.join(GLACIER_PKG, 'ckpt', 'glacier.pt')       # official MassSpecGym weights, re-saved {hyper_parameters, state_dict}
MSPRED_COMMIT = '708148c2a8eb0c32120644436fefd2fe90cd2214'
OPTIONAL = ('h5py', 'multiprocess', 'platformdirs')
_DONE = []


class _Anything:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Anything()
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return _Anything()
    def __bool__(self): return False


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        v = type(k, (_Anything,), {}) if k[:1].isupper() else _Anything()
        setattr(self, k, v)
        return v


class _StubFinder:
    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] not in self.roots:
            return None
        import importlib.machinery as im
        return im.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def setup():
    """Idempotent. Returns ms_pred.common."""
    if _DONE:
        return _DONE[0]
    for p in reversed([os.path.join(HERE, 'shim'), os.path.join(GLACIER_PKG, 'shim'), MSPRED_SRC, ICE_SITE]):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import warnings
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    for name in ('ms_pred.common.plot_utils', 'ms_pred.nn_utils.tune_utils', 'ms_pred.nn_utils.base_hyperopt'):
        sys.modules[name] = types.ModuleType(name)
    for f in ('plot_mol_as_vector', 'plot_compare_ms', 'plot_ms'):
        setattr(sys.modules['ms_pred.common.plot_utils'], f, lambda *x, **k: None)
    missing = []
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
        except BaseException:
            for k in [k for k in sys.modules if k.split('.')[0] == m]:
                del sys.modules[k]
            missing.append(m)
    if missing:
        sys.meta_path.insert(0, _StubFinder(missing))
    import ms_pred.common.chem_utils as cu
    cache_root = os.path.join(tempfile.gettempdir(), 'glacierft_ms_pred_cache')
    cu.user_cache_dir = lambda name, *x, **k: os.path.join(cache_root, name)
    import ms_pred.common as common
    import pygmtools
    assert getattr(pygmtools, '__glacierft_shim__', False), pygmtools.__file__
    _DONE.append(common)
    return common


def load_model(ckpt=None, device='cpu'):
    """Official JointModel from a {hyper_parameters, state_dict} file (strict load). Returns (model, hyper_parameters)."""
    import torch
    setup()
    from ms_pred.glacier import joint_model as jm
    c = torch.load(ckpt or BASE_CKPT, map_location='cpu', weights_only=True)
    hp = {k: v for k, v in dict(c['hyper_parameters']).items() if not str(k).startswith('_')}
    m = jm.JointModel(**hp)
    m.load_state_dict(c['state_dict'], strict=True)
    return m.to(device), hp


def save_model(model, hp, path, meta=None):
    """Same layout as the package checkpoint ckpt/glacier.pt (gl_runner.py --ckpt loads it unchanged)."""
    import torch
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    out = {'hyper_parameters': dict(hp), 'state_dict': sd}
    if meta:
        out['glacierft'] = meta
    tmp = path + '.tmp'
    torch.save(out, tmp)
    os.replace(tmp, path)
