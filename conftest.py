"""
conftest.py
Root-level pytest configuration for NutriSense AI test suite.

Provides:
 - Correct workspace root injection into sys.path.
 - Torch / torchvision mock stubs so tests that only exercise Flask / nutrition
   logic can run on machines where PyTorch is not installed.
 - Shared fixtures available to all test modules.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Workspace root on sys.path so `from app import app` works in every test.
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# 2. Create lightweight mock stubs for heavy ML dependencies if absent.
#    This lets Phase 1-4 / Phase 7 / Label-scanner tests run without GPU libs.
# ---------------------------------------------------------------------------

def _register_ml_mocks():
    """Register ML library mocks only when the real libraries are absent."""
    try:
        import torch  # noqa: F401
        return  # real torch available — no mocking needed
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # torch  (must be a package so sub-imports like torch.utils.data work)
    # ------------------------------------------------------------------
    torch_pkg = types.ModuleType("torch")
    torch_pkg.__path__ = []  # Mark as package

    # torch.device
    class _Device:
        def __init__(self, s="cpu"): self._s = str(s)
        def __str__(self): return self._s
        def __repr__(self): return f"device(type='{self._s}')"

    # torch.Tensor-like stub
    class _Tensor:
        def __init__(self, data=None):
            self.data = data
        def unsqueeze(self, *a): return self
        def squeeze(self, *a): return self
        def dim(self): return 4
        @property
        def shape(self): return (1, 3, 224, 224)
        def to(self, *a, **kw): return self
        def item(self): return 0.0
        def __iter__(self): return iter([self])

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    torch_pkg.device = _Device
    torch_pkg.Tensor = _Tensor
    torch_pkg.no_grad = lambda: _NoGrad()
    torch_pkg.cuda = MagicMock()
    torch_pkg.cuda.is_available = lambda: False
    torch_pkg.load = MagicMock(return_value={"model_state_dict": {}, "epoch": 0})
    torch_pkg.softmax = MagicMock(return_value=_Tensor())
    torch_pkg.topk = MagicMock(return_value=([_Tensor()], [_Tensor()]))
    torch_pkg.zeros = MagicMock(return_value=_Tensor())
    torch_pkg.ones = MagicMock(return_value=_Tensor())

    # torch.nn
    nn_pkg = types.ModuleType("torch.nn")
    nn_pkg.__path__ = []

    class _Module:
        def __init__(self, *a, **kw): pass
        def to(self, *a, **kw): return self
        def eval(self): return self
        def __call__(self, x): return _Tensor()
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])

    nn_pkg.Module = _Module
    nn_pkg.Sequential = MagicMock(return_value=_Module())
    nn_pkg.Linear = MagicMock(return_value=_Module())
    nn_pkg.Dropout = MagicMock(return_value=_Module())
    nn_pkg.CrossEntropyLoss = MagicMock(return_value=_Module())
    torch_pkg.nn = nn_pkg

    # torch.nn.functional
    nn_func = types.ModuleType("torch.nn.functional")
    nn_func.__path__ = []
    torch_pkg.nn.functional = nn_func

    # torch.utils
    utils_pkg = types.ModuleType("torch.utils")
    utils_pkg.__path__ = []
    torch_pkg.utils = utils_pkg

    # torch.utils.data
    class _Dataset:
        def __len__(self): return 0
        def __getitem__(self, idx): return None

    class _DataLoader:
        def __init__(self, dataset, *a, **kw):
            self.dataset = dataset
        def __iter__(self): return iter([])
        def __len__(self): return 0

    data_pkg = types.ModuleType("torch.utils.data")
    data_pkg.__path__ = []
    data_pkg.Dataset = _Dataset
    data_pkg.DataLoader = _DataLoader
    utils_pkg.data = data_pkg

    # Register all torch sub-modules
    sys.modules["torch"] = torch_pkg
    sys.modules["torch.nn"] = nn_pkg
    sys.modules["torch.nn.functional"] = nn_func
    sys.modules["torch.utils"] = utils_pkg
    sys.modules["torch.utils.data"] = data_pkg

    # ------------------------------------------------------------------
    # torchvision
    # ------------------------------------------------------------------
    tv_pkg = types.ModuleType("torchvision")
    tv_pkg.__path__ = []

    # torchvision.transforms
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.__path__ = []

    class _Compose:
        def __init__(self, transforms): self.transforms = transforms
        def __call__(self, x): return x

    tv_transforms.Compose = _Compose
    tv_transforms.Resize = MagicMock()
    tv_transforms.CenterCrop = MagicMock()
    tv_transforms.ToTensor = MagicMock()
    tv_transforms.Normalize = MagicMock()
    tv_transforms.RandomHorizontalFlip = MagicMock()
    tv_transforms.RandomRotation = MagicMock()
    tv_transforms.ColorJitter = MagicMock()
    tv_transforms.RandomResizedCrop = MagicMock()
    tv_pkg.transforms = tv_transforms

    # torchvision.models
    tv_models = types.ModuleType("torchvision.models")
    tv_models.__path__ = []

    class _FakeMobileNetV2:
        def __init__(self, *a, **kw):
            linear = MagicMock()
            linear.in_features = 1280
            self.classifier = [None, linear]
        def to(self, *a, **kw): return self
        def eval(self): return self
        def __call__(self, x): return _Tensor()
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])
        def state_dict(self): return {}

    tv_models.mobilenet_v2 = lambda **kw: _FakeMobileNetV2()
    tv_pkg.models = tv_models

    sys.modules["torchvision"] = tv_pkg
    sys.modules["torchvision.transforms"] = tv_transforms
    sys.modules["torchvision.models"] = tv_models

    # ------------------------------------------------------------------
    # easyocr (used by label_scanner)
    # ------------------------------------------------------------------
    try:
        import easyocr  # noqa: F401
    except ImportError:
        ocr_mock = types.ModuleType("easyocr")
        ocr_mock.__path__ = []

        class _FakeReader:
            def __init__(self, *a, **kw): pass
            def readtext(self, *a, **kw): return []

        ocr_mock.Reader = _FakeReader
        sys.modules["easyocr"] = ocr_mock


_register_ml_mocks()
