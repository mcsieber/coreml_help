"""
Fixtures for pytest'ing ms_core
"""
from pathlib import Path
import pytest
from ..coreml_help import CoremlBrowser
import coremltools.models.utils as cu
import coremltools.models.model as cm

_ml_path = Path('/Volumes/ArielD/storage/data/mlmodels/Test/SqueezeNet.mlmodel')
_ml_compiled_path = Path('/Volumes/ArielD/storage/data/mlmodels/Test/SqueezeNet.mlmodelc')

@pytest.fixture(scope="module")
def ml_path():
  """Return the path to a mlmodel file for testing"""
  return _ml_path


@pytest.fixture(scope="module")
def ml_CoremlBrowser():
  """Return a CoremlBrowser object from a .mlmodel file"""
  cmb   = CoremlBrowser(_ml_path)
  return cmb

@pytest.fixture(scope="module")
def ml_spec_shaper():
  """Return a protobuf specification from a .mlmodel file and a shape inference object"""
  spec   = cu.load_spec(_ml_path)
  nns    = cm.NeuralNetworkShaper(spec)
  return spec,nns


@pytest.fixture(scope="module")
def ml_image_dir():
  """Return the directory containing directories of images"""
  return Path('/Users/mcsieber/github/mcsieber/ms_core/tests/images')