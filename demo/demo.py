# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following atwo lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )


    parser.add_argument( "--save_pred", default=True)
    parser.add_argument( "--save_overlay_withclass", default=True)

    parser.add_argument( "--save_overlay", default=True)
    parser.add_argument( "--save_dilate", default=True)
    parser.add_argument(    "--mask_dilate_kernelsize", type=int, default=5)
    parser.add_argument(    "--mask_dilate_iter",       type=int, default=18)

    parser.add_argument( "--save_complete_mask", default=True)
    parser.add_argument( "--save_component_mask", default=True)
    parser.add_argument(    "--num_component", type=int, default=5)



    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False



def process_and_save_mask(mask_to_save, mask_pathname, mask_overlay_pathname, dilated_mask_pathname, dilated_mask_overlay_pathname):
    # a. Binary Mask
    cv2.imwrite(mask_pathname, mask_to_save*255)

    if args.save_overlay:
        # b. Binary Mask's Overlay
        a = np.expand_dims(mask_to_save, axis=2) # (675, 1200, 1)
        b = np.concatenate((a, a, a), axis=2)*255 # (675, 1200, 3) == img.shape
        overlay = cv2.addWeighted(b,0.75, img,1, 0)
        cv2.imwrite(mask_overlay_pathname, overlay)

 
    if args.save_dilate:
        # c. Dilated Mask 
        # Perform dilation using ellipse kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_dilate_kernelsize, args.mask_dilate_kernelsize))
        dilate_combined_mask = cv2.dilate(mask_to_save.astype(np.uint8), kernel, iterations=args.mask_dilate_iter)
            # To extend the True/1 regions further: increase the kernel size, repeat dilation operation using iteration
        cv2.imwrite(dilated_mask_pathname, dilate_combined_mask*255)

        if args.save_overlay:
            # d. Dilated Mask's Ooverlay
            a = np.expand_dims(dilate_combined_mask, axis=2) # (675, 1200, 1)
            b = np.concatenate((a, a, a), axis=2)*255 # (675, 1200, 3) == img.shape
            dilated_overlay = cv2.addWeighted(b,0.75, img,1, 0)
            cv2.imwrite(dilated_mask_overlay_pathname, dilated_overlay)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            if False:
                pass
                # print(f"predictions['instances'].pred_boxes.tensor.cpu().numpy()={predictions['instances'].pred_boxes.tensor.cpu().numpy()}")
                    # predictions, visualized_output = demo.run_on_image(img) 
                    # printing predictions:
                    # {'instances': Instances(num_instances=2, image_height=480, image_width=480, 
                    #         fields=[
                    #                 pred_boxes: Boxes(tensor([[ 86.8838, 276.6702, 438.8151, 374.7910],
                    #                         [3.9292, 169.9821, 124.6545, 286.7554]], device='cuda:0')), 
                    #                 scores: tensor([0.9835, 0.8310], device='cuda:0'), 
                    #                 pred_classes: tensor([66, 56], device='cuda:0'), 
                    #                 pred_masks: tensor([[[False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 ...,
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False]],

                    #                 [[False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 ...,
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False],
                    #                 [False, False, False,  ..., False, False, False]]], device='cuda:0')
                    #         ]
                    #         )        
                    # }

                    # type(predictions)=<class 'dict'>
                    # type(predictions['instances'])=<class 'detectron2.structures.instances.Instances'>
                    # type(predictions['instances'].pred_boxes)=<class 'detectron2.structures.boxes.Boxes'>
                    # predictions['instances'].pred_boxes=Boxes(
                                # tensor([[ 86.8838, 276.6702, 438.8151, 374.7910],
                                #         [  3.9292, 169.9821, 124.6545, 286.7554]], 
                                #        device='cuda:0')
                                # )
                    # predictions['instances'].pred_boxes.tensor.cpu().numpy()=[[ 86.88382   276.6702    438.81512   374.79102  ]
                            #  [  3.9291596 169.98207   124.65453   286.75543  ]
                            #  [352.12546   143.4035    478.20728   280.10855  ]
                            #  [  1.3419663 271.6118     94.49982   380.61316  ]
                            #  [421.9968    282.93454   480.        385.5607   ]
                            #  [ 28.962715   49.783978   99.72068    74.628815 ]
                            #  [ 39.62111   109.77912   107.24372   137.54713  ]], shape=(num_predictions, 4)
                
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if (len(args.input) == 1): 
                    print("The modified code only supports directory of data input")
                    exit()
                    if os.path.isdir(args.output):
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        out_filename = args.output

                else:
                    if os.path.isdir(args.output):
                        pass
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output" # destined to raise assertion error


                fname, ext = os.path.basename(path).split('.')[0], os.path.basename(path).split('.')[1]

                # 1. Save prediction (contains: pred_boxes)
                if args.save_pred:
                    pred_dirpath = os.path.join(args.output, "pred")
                    if not os.path.exists(pred_dirpath): os.makedirs(pred_dirpath)
                    np.savetxt(os.path.join(pred_dirpath, f"{fname}_pred_boxes.txt"), 
                               predictions['instances'].pred_boxes.tensor.cpu().numpy()) # other options: scores, pred_classes, pred_masks
                        # a = np.loadtxt('indoor_dataset_2/0803out_4_thres001/pred/clean_1_pred_boxes.txt') #(100,4)
                        # with open(json_filename, "w") as fp:  json.dump(predictions, fp)  # encode dict into JSON

                # 2. Save overlayed image
                if args.save_overlay_withclass:
                    mask_overlay_dirpath = os.path.join(args.output, "overlay_withclass")
                    if not os.path.exists(mask_overlay_dirpath): os.makedirs(mask_overlay_dirpath)
                    visualized_output.save(os.path.join(mask_overlay_dirpath, f"{fname}_overlay_withclass.{ext}"))


                # 3. Save binary mask (merged boxes), mask overlay, dilated mask, dilated mask overlay
                masks = predictions['instances'].get('pred_masks').to('cpu').numpy()
                num, h, w = masks.shape
                combined_masks, mask_pathnames, dilated_mask_pathnames, dilated_overlay_pathnames = [], [], [], []
                # args.mask_dilate_kernelsize, args.mask_dilate_iter = int(args.mask_dilate_kernelsize), int(args.mask_dilate_iter) # TODO: remove


                ## 3a. complete: one overall mask
                if args.save_complete_mask:
                    mask_dirpath =                 os.path.join(args.output, "complete_mask", "mask")
                    mask_overlay_dirpath =         os.path.join(args.output, "complete_mask", "mask_overlay")
                    dilated_mask_dirpath =         os.path.join(args.output, "complete_mask", f"dilated_mask_kernel{args.mask_dilate_kernelsize}iter{args.mask_dilate_iter}")
                    dilated_mask_overlay_dirpath = os.path.join(args.output, "complete_mask", f"dilated_mask_kernel{args.mask_dilate_kernelsize}iter{args.mask_dilate_iter}_overlay")
                    if not os.path.exists(mask_dirpath): os.makedirs(mask_dirpath)
                    if not os.path.exists(mask_overlay_dirpath): os.makedirs(mask_overlay_dirpath)
                    if not os.path.exists(dilated_mask_dirpath): os.makedirs(dilated_mask_dirpath)
                    if not os.path.exists(dilated_mask_overlay_dirpath): os.makedirs(dilated_mask_overlay_dirpath)
                    print(f"...saving complete masks to: {os.path.join(args.output, 'complete_mask')}")

                    complete_mask = np.zeros((h, w))
                    for m in masks: 
                        complete_mask = np.logical_or(complete_mask, m)
                    process_and_save_mask(complete_mask, 
                                         os.path.join(mask_dirpath,         f"{fname}_mask.{ext}"), os.path.join(mask_overlay_dirpath,         f"{fname}_overlay.{ext}"),
                                         os.path.join(dilated_mask_dirpath, f"{fname}_mask.{ext}"), os.path.join(dilated_mask_overlay_dirpath, f"{fname}_overlay.{ext}"))

                    
                ## 3b: multiple component masks
                if args.save_component_mask:
                    component_mask_dirpath =                 os.path.join(args.output, "component_mask", "mask")
                    component_mask_overlay_dirpath =         os.path.join(args.output, "component_mask", "mask_overlay")
                    component_dilated_mask_dirpath =         os.path.join(args.output, "component_mask", f"dilated_mask_kernel{args.mask_dilate_kernelsize}iter{args.mask_dilate_iter}")
                    component_dilated_mask_overlay_dirpath = os.path.join(args.output, "component_mask", f"dilated_mask_kernel{args.mask_dilate_kernelsize}iter{args.mask_dilate_iter}_overlay")
                    if not os.path.exists(component_mask_dirpath): os.makedirs(component_mask_dirpath)
                    if not os.path.exists(component_mask_overlay_dirpath): os.makedirs(component_mask_overlay_dirpath)
                    if not os.path.exists(component_dilated_mask_dirpath): os.makedirs(component_dilated_mask_dirpath)
                    if not os.path.exists(component_dilated_mask_overlay_dirpath): os.makedirs(component_dilated_mask_overlay_dirpath)
                    print(f"...saving component masks to: {os.path.join(args.output, 'component_mask')}")

                    component_size = num//args.num_component # eg. 76//5=15
                    start_indices = list(np.arange(args.num_component)*component_size) # eg. [0, 15, 30, 45, 60]
                    start_indices.append(num) # eg. [0, 15, 30, 45, 60, 76]
                    
                    for i in range(args.num_component): #==len(start_indices)-1)
                        component_mask = np.zeros((h, w))
                        for m in masks[start_indices[i] : start_indices[i+1]]: # last component may have more boxes (eg. 16 for 76 total)
                            component_mask = np.logical_or(component_mask, m)
                        process_and_save_mask(component_mask, 
                                              os.path.join(component_mask_dirpath,         f"{fname}_mask{i}.{ext}"), os.path.join(component_mask_overlay_dirpath,         f"{fname}_overlay{i}.{ext}"),
                                              os.path.join(component_dilated_mask_dirpath, f"{fname}_mask{i}.{ext}"), os.path.join(component_dilated_mask_overlay_dirpath, f"{fname}_overlay{i}.{ext}"))
 
                        combined_masks.append(component_mask)
                        mask_pathnames.append(os.path.join(component_mask_dirpath, f"{fname}_mask{i}.{ext}"))
                        dilated_mask_pathnames.append(os.path.join(component_dilated_mask_dirpath, f"{fname}_mask{i}.{ext}"))
                        dilated_overlay_pathnames.append(os.path.join(component_dilated_mask_overlay_dirpath, f"{fname}_dilated_overlay{i}.{ext}"))

                    # print(combined_mask)
                    # print(f"np.sum(combined_mask)={np.sum(combined_mask)}", h, w)
                    


            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit





    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
