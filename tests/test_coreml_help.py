
from coreml_help import *
from re import search

"""
Tests for ms_core
"""


def test_CoremlBrowser(ml_path):
  """ Test creation of the browser object"""
  ml_file = ml_path
  assert ml_file.exists()
  assert ml_file.is_file()
  cmb = CoremlBrowser(ml_file)
  assert cmb.mf_path is not None
  assert cmb.spec is not None
  assert cmb.shaper is not None
  assert cmb.nn is not None
  assert cmb.layers is not None
  assert cmb.layer_count >= 2
  assert cmb.layer_dict is not None
  assert cmb.name_len_centile > 1


def test_get_nn(ml_CoremlBrowser):
  cmb = ml_CoremlBrowser
  nn = cmb.get_nn()
  assert nn is not None
  assert nn.layers is not None
  assert len(nn.layers) >= 2
  assert nn.layers[0] is not None
  assert nn.layers[0].name is not None
  assert nn.layers[0].input is not None
  assert nn.layers[0].input[0] is not None
  assert nn.layers[0].output is not None
  assert nn.layers[0].output[0] is not None
 # self.spec = spec


def _check_show_nn(capsys):
  """
  Common sanity checks on the output of a show_nn
  (splits on blank lines, so that the lines correspond to layers,
  regardless layers formatted for 1 or 2 lines)

  - Check that expected header strings are present
  - Check that expected layer formatting is present

    Return:
      The array of lines, so that the caller can make more specific checks if desired
  """
  out       = capsys.readouterr().out
  ly        = out.split("\n\n")
  n_layers  = len(ly)
  header    = ly[0]
  line1     = ly[1]
  assert n_layers >= 2
  assert search('Stride',header)
  assert search(r"^\s+\d+", line1)
  assert search(r"\[", line1)
  assert search(r"\]", line1)
  return ly


def test_show_nn_default(ml_CoremlBrowser,capsys):
  """
  Sanity check the output a show_nn
  (splits on blank lines, so that the lines correspond to layers,
  regardless layers formatted for 1 or 2 lines)
  Check that expected header strings are present
  Check that expected layer info is present
  """
  cmb = ml_CoremlBrowser
  cmb.show_nn()
  _check_show_nn(capsys)

def test_show_nn_0_5(ml_CoremlBrowser,capsys):
  """
  Verify that there are 5 lines in the output
  (if there are 5 layers ...)
  """
  count = 5
  cmb = ml_CoremlBrowser
  cmb.show_nn(0,count)
  lines = _check_show_nn(capsys)

  if  cmb.layer_count >= count :
      assert len(lines) == count+1
      assert search(r"^\s+1",lines[1])


def test_show_nn_shapes(ml_CoremlBrowser, capsys):
  """ Check for a 'shape' or 'key not found' """
  cmb = ml_CoremlBrowser
  cmb.show_nn(0,5)
  lines     = _check_show_nn(capsys)
  l1        = lines[1]
  assert search("CHW=",l1) or search('key not found',l1)


def test_show_nn_neg(ml_CoremlBrowser, capsys):
  """ Verify that negative idxs work"""
  cmb = ml_CoremlBrowser
  cmb.show_nn(-3)
  lines     = _check_show_nn(capsys)
  last_line = lines[-2]
  num       = len(cmb.layers)-1
  assert search(f"{num}",last_line)


def test_get_rand_images(ml_image_dir):
  print(ml_image_dir)
  rand_images = get_rand_images(ml_image_dir, n_images=4, search_limit=400)
  assert len(rand_images) > 0
  for f in rand_images: assert is_imgfile(f)
  print (rand_images)


def test_connect_layers(ml_CoremlBrowser):

  cmb  = ml_CoremlBrowser
  nnl = cmb.layers

  # Skip layer 2
  layer1_name = nnl[1].name
  layer3_name = nnl[3].name

  changes = cmb.connect_layers(from_=layer1_name, to_=layer3_name)

  assert changes.error is None
  assert changes.changed_layer == layer3_name
  assert nnl[1].output[0] == nnl[3].input[0]
  print(changes)


def test_connect_layers_error(ml_CoremlBrowser):

  from copy import deepcopy

  cmb  = ml_CoremlBrowser
  nnl = cmb.layers

  # Skip layer 2
  layer1_name   = "xyzzy_1"
  layer3_name   = "xyzzy_3"
  layer3_output = deepcopy(nnl[3].input[0])

  changes = cmb.connect_layers(from_=layer1_name, to_=layer3_name)

  assert changes.changed_layer == 'NONE'
  assert changes.input_before is None
  assert changes.input_after is None
  assert nnl[3].input[0] == layer3_output
  assert changes.error is not None
  assert changes.error.index(layer1_name)
  assert changes.error.index(layer3_name)
  print(changes)



def test_delete_layers(ml_CoremlBrowser):

  cmb    = ml_CoremlBrowser
  nnl   = cmb.layers

  nn_len_before = len(nnl)
  layer2_name   = nnl[2].name
  layer3_name   = nnl[3].name
  bogus_name    = 'xyzzy'
  deleted       = cmb.delete_layers([layer2_name, layer3_name, bogus_name])
  deleted_names = [ d['deleted_layer'] for d in deleted ]

  assert len(nnl) == nn_len_before - 2
  assert layer2_name in deleted_names
  assert layer3_name in deleted_names
  assert bogus_name not in deleted_names
  assert nnl[2].name != layer2_name
  assert nnl[3].name != layer3_name
  print(deleted)

