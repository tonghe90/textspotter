这是我们用来做文字检测实验的caffe，增加了所需的layer，训练脚本在 caffe/example/text_detect，主要由四个:

- train_pvanet.pt: 使用FCN方法做文字检测，目前我们做的比较work，使用ICDAR15的1000张训练数据训练，测试F-score应在84%以上
- train_2stage.pt: 在上一个的基础上加上了第二个stage来refine分类和回归的结果，没有做work，相比第一个没有提升
- train_pvanet_deform.pt: 将第一个网络的部分卷积层使用deform conv进行替换，没有做太多实验，比第一个略有提升，这个可以做更多实验
- e2e_best.pt: 文字检测与识别同时训练，一般先训练识别然后再一起训练，对检测有提升，使用ICDAR15的1000张训练数据训练，测试F-score应在85%以上，识别相比单独训练较差，可以做更多尝试看看能不能得到更好结果

测试的脚本在 caffe/python/src 中:

- unitbox_test_old.py: 测试文字检测模型，对应上面第一，三模型
- e2e_test.py: 测试检测识别end to end，对应上面第四个模型
- mAP.py: 计算mAP

有问题请随时问我，谢谢