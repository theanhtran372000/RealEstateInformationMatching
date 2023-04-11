# Prepare system path
import os
import sys

sys.path.append(os.path.abspath(os.path.join('PaddleOCR', 'tools', 'infer')))
sys.path.insert(0, os.path.abspath('PaddleOCR'))

sys.path.insert(0, os.path.abspath('VietnameseOcrCorrection'))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# Import libs
import cv2
import time
import sys
import yaml
import json
import openai
from PIL import Image
import numpy as np

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

from utils import box_crop_and_save, vertical_crop, box_crop
from utils import generate_prompt, get_chatgpt_answer
from utils import merge_boxes
from utils import VietOCR
# from utils import OCRCorrection

logger = get_logger()

class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                    [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh

            if args.det_box_type == 'poly':
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3

        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_box_type
            postprocess_params["scale"] = args.det_pse_scale
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_box_type
        elif self.det_algorithm == "CT":
            pre_process_list[0] = {'ScaleAlignedShort': {'short_size': 640}}
            postprocess_params['name'] = 'CTPostProcess'
        else:
            logger.info("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'det', logger)

        if self.use_onnx:
            img_h, img_w = self.input_tensor.shape[2:]
            if img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
                pre_process_list[0] = {
                    'DetResizeForTest': {
                        'image_shape': [img_h, img_w]
                    }
                }
        self.preprocess_op = create_operators(pre_process_list)

        if args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="det",
                model_precision=args.precision,
                batch_size=1,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=2,
                logger=logger)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        if self.args.benchmark:
            self.autolog.times.start()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        if self.args.benchmark:
            self.autolog.times.stamp()
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if self.args.benchmark:
                self.autolog.times.stamp()

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs[0]
            preds['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs[0]
            preds['f_score'] = outputs[1]
            preds['f_tco'] = outputs[2]
            preds['f_tvo'] = outputs[3]
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs[0]
        elif self.det_algorithm == 'FCE':
            for i, output in enumerate(outputs):
                preds['level_{}'.format(i)] = output
        elif self.det_algorithm == "CT":
            preds['maps'] = outputs[0]
            preds['score'] = outputs[1]
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']

        if self.args.det_box_type == 'poly':
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        if self.args.benchmark:
            self.autolog.times.end(stamp=True)
        et = time.time()
        return dt_boxes, et - st


if __name__ == "__main__":
    args = utility.parse_args()
    
    # Prepare env
    logger.info('Prepare environment')
    image_file_list = get_image_file_list(args.image_dir)
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    
    # Read configs
    with open('configs.yml', 'r') as f:
        configs = yaml.load(f, yaml.FullLoader)
    
    # Config ChatGPT
    openai.api_key = configs['openai']['api_key']
    
    # Prepare model
    logger.info('Prepare model')
    text_detector = TextDetector(args)
    predictor = VietOCR(device=configs['vietocr']['device'])
    # corrector = OCRCorrection(
    #     configs['ocr_correction']['weight_path'],
    #     configs['ocr_correction']['model_type'],
    #     configs['ocr_correction']['device']
    # )
    
    # Warmup model
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)
    
    # Loop through all test images
    total_time = 0
    save_results = []
    for idx, image_file in enumerate(image_file_list):
        
        # Check image file is multipage or not
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
            
            # === Crop image === #
            l_img, r_img = vertical_crop(img)
            
        if not flag_pdf:
            if l_img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [l_img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
            
        # Foreach page
        for index, img in enumerate(imgs):
            
            # === Text Detection === #
            # Detect boxes
            st = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - st
            total_time += elapse
            
            # Log results
            # logger.info(save_pred)
            if len(imgs) > 1:
                logger.info("{}_{} The predict time of {}: {}".format(
                    idx, index, image_file, elapse))
            else:
                logger.info("The predict time of {}: {:.2f}s".format(image_file, elapse))
            
            logger.info('Found {} boxes!'.format(len(dt_boxes)))
            
            # Draw boxes on image
            src_im = utility.draw_text_det_res(dt_boxes, img.copy())
            
            # Save images
            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace('.pdf',
                                               '_' + str(index) + '.png')
            else:
                save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir,
                "det_raw_{}".format(os.path.basename(save_file)))
            cv2.imwrite(img_path, src_im)
            logger.info("The visualized image saved in {}".format(img_path))
            
            # === Postprocess boxes === #
            # Merge boxes
            dt_boxes, lines = merge_boxes(dt_boxes)
            
            logger.info('Merge down to {} lines, contains {} boxes!'.format(len(lines), len(dt_boxes)))
            
            # Save results
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + str(
                        json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + str(
                    json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            save_results.append(save_pred)
            
            # Crop text
            crop_text_save_dir = os.path.join(draw_img_save_dir, 'subimgs')
            os.makedirs(crop_text_save_dir, exist_ok=True)
            box_crop_and_save(img, dt_boxes, crop_text_save_dir)
            logger.info('All sub-images saved to {}!'.format(crop_text_save_dir))
            
            # Draw boxes on image
            src_im = utility.draw_text_det_res(dt_boxes, img.copy())
            
            # Save images
            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace('.pdf',
                                               '_' + str(index) + '.png')
            else:
                save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir,
                "det_merged_{}".format(os.path.basename(save_file)))
            cv2.imwrite(img_path, src_im)
            logger.info("The visualized image saved in {}".format(img_path))
            
            # === OCR === #
            # Perform OCR to extract text
            logger.info('Extracting text using VietOCR ...')
            since = time.time()
            all_texts = []
            for i, line in enumerate(lines):
                
                text_in_line = []
                for box in line:
                    subimg = box_crop(img, box)
                    subimg = Image.fromarray(subimg)
                    
                    t = predictor(subimg)
                    
                    text_in_line.append(t)
                
                line_text = ' '.join(text_in_line)
                all_texts.append(line_text)
                logger.info('[{}] Line: {}'.format(i + 1, line_text))
            
            raw_text = '\n'.join(all_texts)
            text_path = os.path.join(draw_img_save_dir, 'text_raw.txt')
            with open(text_path, 'w') as f:
                f.write(raw_text)
            logger.info('Done after {:.2f}s!'.format(time.time() - since))
            logger.info('Raw text: \n {}'.format(raw_text))
            logger.info('Raw texts saved to {}!'.format(text_path))
            
#             # OCR Correction with Seq2Seq
#             since = time.time()
            
#             corrected_texts = []
#             for i, text in enumerate(all_texts):
#                 out = corrector(text)
#                 corrected_texts.append(out)
#                 logger.info('[{}] Corrected line: {}'.format(i + 1, out))
            
#             final_text = '\n'.join(corrected_texts)
#             logger.info('Done after {:.2f}s!'.format(time.time() - since))
#             logger.info('Final text: \n {}'.format(final_text))
            
            # === OCR Correction with ChatGPT === #
            since = time.time()
            logger.info('Correcting text using ChatGPT ...')
            correction_prompt = generate_prompt(configs['prompt']['correction'], raw_text)
            final_text = get_chatgpt_answer(correction_prompt)
            logger.info('Done after {:.2f}s!'.format(time.time() - since))
            logger.info('Final text: \n {}'.format(final_text))
            
            # Save text
            text_path = os.path.join(draw_img_save_dir, 'text_final.txt')
            with open(text_path, 'w') as f:
                f.write(final_text)
            logger.info('Final texts saved to {}!'.format(text_path))
            
            # === Information extraction with ChatGPT === #
            since = time.time()
            logger.info('Extracting information using ChatGPT ...')
            extraction_prompt = generate_prompt(configs['prompt']['extraction'], final_text)
            extracted_infos = get_chatgpt_answer(extraction_prompt)
            logger.info('Done after {:.2f}s!'.format(time.time() - since))
            logger.info('Information: \n {}'.format(extracted_infos))
            
            # Save text
            text_path = os.path.join(draw_img_save_dir, 'infos.json')
            with open(text_path, 'w') as f:
                f.write(extracted_infos)
            logger.info('Extracted information saved to {}!'.format(text_path))
            
            # === Real estate shape extraction === #
            # ...

    with open(os.path.join(draw_img_save_dir, "det_results.txt"), 'w') as f:
        f.writelines(save_results)
        f.close()
    if args.benchmark:
        text_detector.autolog.report()