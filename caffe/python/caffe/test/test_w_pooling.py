import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

N = 4
C = 4
H = 4
W = 4
for i in range(1):
	# set the inputs
	input_names_and_values = [('lstm_output', np.random.rand(N,C,H,W)), ('cont', -np.random.rand(W,N)*W)]
	output_names = ['out1']
	py_module = 'layers.w_pooling'
	py_layer = 'WPoolingLayer'
	param_str = "'pooling_method': ave"
	propagate_down = [True, False]
	bp_top = [True]

	# call the test
	test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down, bp_top)

# you are done!