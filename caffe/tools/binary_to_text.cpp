#include <string>
#include <tr1/unordered_map>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

/*******************************************
******  convert binary prototxt to text  ***
******  author: Liang Ding  ****************
******  email: liangding@sensetime.com *****
******  date: 2016.9.2  ********************
*******************************************/

using namespace caffe;

typedef std::tr1::unordered_map<std::string, std::string> umap;
NetParameter remove_blobs(NetParameter &src_param);
NetParameter remove_mpi(NetParameter &src_param);

int main(int argc, char** argv) {
	if (argc < 3 || argc > 5) {
		LOG(ERROR) << "Usage: "
							 << "binary_to_text net_proto_file_in net_proto_file_out [--remove_blobs] [--remove_mpi]";
		return 1;
	}

	NetParameter net_param;
	ReadNetParamsFromBinaryFileOrDie(argv[1], &net_param);
	if (argc == 5)
		net_param = remove_mpi(net_param);
	if (argc >= 4)
		net_param = remove_blobs(net_param);

	WriteProtoToTextFile(net_param, argv[2]);
	LOG(INFO) << "Wrote NetParameter text proto to " << argv[2];
	return 0;
}

NetParameter remove_blobs(NetParameter &src_param) {
	NetParameter dst_param;
	umap c_map;
	for (int i = 0; i < src_param.layer_size(); ++i) {
		LayerParameter *layer = src_param.mutable_layer(i);
		if (layer->type() == "Split") {
			for (int j = 0; j < layer->top_size(); ++j)
				c_map[layer->top(j)] = layer->bottom(0);
		} else {
			layer->clear_blobs();
			layer->clear_phase();
			for (int j = 0; j < layer->bottom_size(); ++j)
				if (c_map.find(layer->bottom(j)) != c_map.end())
					layer->set_bottom(j, c_map[layer->bottom(j)]);
			dst_param.add_layer()->CopyFrom(*layer);
		}
	}
	return dst_param;
}

NetParameter remove_mpi(NetParameter &src_param) {
	NetParameter dst_param;
	umap c_map;
	for (int i = 0; i < src_param.layer_size(); ++i) {
		LayerParameter *layer = src_param.mutable_layer(i);
		string *layer_type = layer->mutable_type();
		if (*layer_type == "Gather" || *layer_type == "Scatter") {
			for (int j = 0; j < layer->top_size(); ++j)
				c_map[layer->top(j)] = layer->bottom(j);
		} else {
			for (int j = 0; j < layer->bottom_size(); ++j)
				if (c_map.find(layer->bottom(j)) != c_map.end())
					layer->set_bottom(j, c_map[layer->bottom(j)]);
			if (layer->bottom_size() && layer_type->rfind("Data") == (layer_type->size() - 4) )
				layer_type->erase(layer_type->size() - 4, 4);
			dst_param.add_layer()->CopyFrom(*layer);
		}
	}
	return dst_param;
}