#include <string>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

/*******************************************
******  copy / remove layers  **************
******  author: Liang Ding  ****************
******  email: liangding@sensetime.com *****
******  date: 2017.1.17  *******************
*******************************************/

using namespace caffe;

int main(int argc, char** argv) {
	if (argc != 4) {
		LOG(ERROR) << "Usage: "
		           << "copy_layers weights mask net_proto_file_out";
		return 1;
	}

	NetParameter net_param, mask;
	ReadNetParamsFromBinaryFileOrDie(argv[1], &net_param);
	ReadNetParamsFromTextFileOrDie(argv[2], &mask);
	std::map<std::string, int> layer_names;
	for (int i = 0; i < net_param.layer_size(); ++i)
		layer_names[net_param.layer(i).name()] = i;

	for (int i = 0; i < mask.layer_size(); ++i) {
		const std::string layer_name = mask.layer(i).name();
		if (layer_names.find(layer_name) != layer_names.end()) {
			auto &blobs = net_param.layer(layer_names[layer_name]).blobs();
			for (auto &b : blobs)
				mask.mutable_layer(i)->add_blobs()->CopyFrom(b);
		}
	}
	WriteProtoToBinaryFile(mask, argv[3]);
	return 0;
}

