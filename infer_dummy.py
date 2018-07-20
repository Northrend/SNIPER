# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
import os
import json
import time
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from symbols.faster import *
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--cats', dest='cats_file', help='Path to the cats file',
    							default='',type=str)
    arg_parser.add_argument('--model_prefix', dest='model_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--model_epoch', dest='model_epoch', help='Epoch of model',
                            default=0, type=int)
    arg_parser.add_argument('--single', dest='single_image_test', help='Set to test one single image', type=bool, 
                            default=False)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image or image list', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--im_prefix', dest='im_prefix', help='Prefix of the image path', type=str,
                            default='')
    arg_parser.add_argument('--vis', dest='vis', help='Set to visualize', type=bool,
                            default=False)
    arg_parser.add_argument('--thresh', dest='thresh', help='Threshold', type=float,
                            default=0.7)
    arg_parser.add_argument('--output_json', dest='output_json', help='Path to output json file',
                            default='tmp.json', type=str)
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


def main():
    tic_0 = time.time()
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    # Use just the first GPU for demo
    context = [mx.gpu(int(config.gpus[0]))]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Imput images
    if args.single_image_test:
        # Get image dimensions
        width, height = Image.open(args.im_path).size
        # Pack image info
        roidb = [{'image': args.im_path, 'width': width, 'height': height, 'flipped': False}]
    else:
        roidb = list()
        with open(args.im_path, 'r') as f: 
            image_list = [x.strip() for x in f.readlines()]
        for image_tmp in image_list:
            im_path = os.path.join(args.im_prefix, image_tmp) if args.im_prefix else image_tmp
            # Get image dimensions
            width, height = Image.open(im_path).size
            # Pack image info
            roidb.append({'image': im_path, 'width': width, 'height': height, 'flipped': False})

    # Creating the Logger
    # logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # Pack db info
    db_info = EasyDict()
    db_info.name = 'dummy'
    db_info.result_path = 'data/demo'

    with open(args.cats_file, 'r') as f:
        db_info.classes = ['__background__']
        for buff in f.readlines():
            db_info.classes.append(buff.strip().split(',')[-1])
    print('=> categories: {}'.format(db_info.classes))
    db_info.num_classes = len(db_info.classes)

    # Create the model
    sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
    sym_inst = sym_def(n_proposals=400, test_nbatch=1)
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                               crop_size=None, test_scale=config.TEST.SCALES[0],
                               num_classes=db_info.num_classes)
    # Create the module
    shape_dict = dict(test_iter.provide_data_single)
    sym_inst.infer_shape(shape_dict)
    mod = mx.mod.Module(symbol=sym,
                        context=context,
                        data_names=[k[0] for k in test_iter.provide_data_single],
                        label_names=None)
    mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

    # Initialize the weights
    arg_params, aux_params = load_param(args.model_prefix, args.model_epoch,
                                        convert=True, process=True)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # Create the tester
    tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=1)

    tic_1 = time.time()
    # Sequentially do detection over scales
    # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
    all_detections= []
    for s in config.TEST.SCALES:
        # Set tester scale
        tester.set_scale(s)
        # Perform detection
        all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

    tic_2 = time.time()
    # Aggregate results from multiple scales and perform NMS
    tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
    file_name, out_extension = os.path.splitext(os.path.basename(args.im_path))
    all_detections = tester.aggregate(all_detections, vis=args.vis, cache_name=None, vis_path='./data/demo/',
                                          vis_name='{}_detections'.format(file_name), vis_ext=out_extension)
    # pprint.pprint(all_detections)
    # print(len(all_detections))

    tic_3 = time.time()
    # genrate json result
    result = dict()
    for idx,img in enumerate(image_list): 
        img_name = os.path.basename(img)
        result[img_name] = list()
        for cat_idx in range(1,db_info.num_classes):
            bboxes = all_detections[cat_idx][idx]
            for box in bboxes:
                score = box[4]
                if score >= args.thresh:
                    result[img_name].append([round(float(x),6) for x in box] + [db_info.classes[cat_idx]])
    with open(args.output_json,'w') as f:
        json.dump(result,f,indent=2)
    print('=> Time Cost:\nInitialization: {:.4f}s\nInference: {:.4f}s\nPost processing: {:.4f}s\nGenerate result: {:.4f}s\nTotal: {:.4f}s'.format(tic_1-tic_0, tic_2-tic_1, tic_3-tic_2, time.time()-tic_3, time.time()-tic_0))
    return all_detections

if __name__ == '__main__':
    main()
