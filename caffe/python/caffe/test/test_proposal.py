import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

N = 4
H = 4
W = 4
for i in range(10):
	# set the inputs
	input_names_and_values = [('score_4s_softmax', np.random.rand(N,2,H,W)), ('pre_bbox_orient', np.random.rand(N,5,H,W) + 1), ('gt_bbox', np.random.rand(N,1,5,9)), ('ignore_bbox', np.random.rand(N,1,6,9))]
	output_names = ['out1', 'out2', 'out3', 'out4']
	py_module = 'layers.proposal'
	py_layer = 'ProposalLayer'
	param_str = "'max_w': 128 \n'phase': 1 \n'threshold': 0.7 \n'bbox_scale': 1 \n'num_proposal': 16  \n'fg_iou': 0.7 \n'bg_iou': 0.3 \n'ignore_iou': 0.3 \n'out_height': 8 \n'nms_score': 0.15 \n'scale': 0.125"
	propagate_down = [False, True, False, False]
	bp_top = [False, False, True, False]

	# call the test
	test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down, bp_top)

# you are done!