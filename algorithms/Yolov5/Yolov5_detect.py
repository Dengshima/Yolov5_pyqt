import argparse

import torch.backends.cudnn as cudnn

# from .utils import google_utils
from utils.datasets import *
from utils.utils import *
def nms_class(test):
    import torch
    thresh=0.85
    # print('before nms:', test[0].shape)
    test=test[0]
    if test==None:
       return [test]
    row=int(test.numel()/6)#计算行数
    #print(test)
    line=[i for i in range(0,row)]#初始化列表数
    for i in range(row-1):
        x1=float(test[i][0])
        y1=float(test[i][1])
        w1=float(test[i][2])
        h1=float(test[i][3])
        conf1=float(test[i][4])
        s1=w1*h1#读出xywh计算s
        
        for j in range(i+1,row):
            x2=float(test[j][0])
            y2=float(test[j][1])
            w2=float(test[j][2])
            h2=float(test[j][3])
            conf2=float(test[j][4])
            s2=w2*h2#读出xywh计算s
            top=min(y1+h1/2,y2+h2/2)
            bottom=max(y1-h1/2,y2-h2/2)
            left=max(x1-w1/2,x2-w2/2)
            right=min(x1+w1/2,x2+w2/2)#计算四角
            if left>= right or top <= bottom:
                iou=0
            else:
                inter=(right-left)*(top-bottom)
                iou=inter/(s1+s2-inter)
            if iou>thresh:
                if conf1>conf2:
                    delete=j
                else:
                    delete=i
                if delete in line :#删除
                    line.remove(delete)

    #line_t=torch.tensor(line)
    #test_new=torch.index_select(test,0,line_t) #can not index_select
    # device0 = torch_utils.select_device('')
    # cuda0 = torch.device('cuda:0')
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred=torch.rand([len(line), 6], device=device0)
    for i in range(len(line)):
        for j in range(6):
            #print(i,j,pred[i][j],test,list)
            pred[i][j]=float(test[line[i]][j])
            #pred[i][j]=0.3
    test_new=[pred]
    print('after nms:', pred.shape)
    return test_new

def detect(source_image, model_weight, output='inference/output', save_img=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=model_weight, help='model.pt path')
    parser.add_argument('--source', type=str, default=source_image, help='source')
    # parser.add_argument('--source', type=str, default='test_img/0001.JPEG', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=output, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)
    
    result_images = []
    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred=nms_class(pred)
        #print(pred,pred[0].shape)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    #print(s)

                # Write results
                if save_txt:
                    with open(txt_path + '.txt', 'a') as f:
                         f.write(s+ '\n')
                       
                    
                
                for *xyxy, conf, cls in det:
                    #print (det)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls,conf, *xywh))  # label format
                            #f.write(s+ '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(0) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    result_images.append(save_path)
                    cv2.imwrite(save_path, im0)
                    # print(type(im0))
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return result_images


if __name__ == '__main__':
    with torch.no_grad():
        detect('test_img/RGBs/0001.JPEG', 'weights/RGBbest.pt')

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
