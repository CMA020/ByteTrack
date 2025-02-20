import cv2
import numpy as np
import torch
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.exp import get_exp
from yolox.utils import fuse_model

def cvt_preprocessing(img, test_size, test_conf):
    """Preprocess image for YOLOX input"""
    if len(img.shape) == 3:
        padded_img = np.ones((test_size[0], test_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(test_size, dtype=np.uint8) * 114
    
    r = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img /= 255.0
    
    return torch.from_numpy(padded_img).unsqueeze(0), r

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """Postprocess YOLOX outputs"""
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = cv2.dnn.NMSBoxes(
            detections[:, :4].cpu().numpy(),
            detections[:, 4].cpu().numpy(),
            conf_thre,
            nms_thre,
        )

        detections = detections[nms_out_index]
        output[i] = detections

    return output

class LiveStreamTracker:
    def __init__(self, model_path, exp_file):
        # Initialize YOLOX detector
        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()
        self.model.eval()
        
        # Load model weights
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model)
        
        class TrackerArgs:
            def __init__(self):
                self.track_thresh = 0.5    # Base threshold for detection
                self.track_buffer = 30     # Track buffer size
        
        # Initialize ByteTracker with args object and frame_rate
        self.tracker = BYTETracker(
            args=TrackerArgs(),  # Pass args object with required attributes
            frame_rate=30        # Default frame rate
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
    
    def preprocess(self, frame):
        img, ratio = cvt_preprocessing(
            frame,
            self.exp.test_size,
            self.exp.test_conf
        )
        if torch.cuda.is_available():
            img = img.cuda()
        return img, ratio
    
    def detect_and_track(self, frame):
        img, ratio = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.exp.num_classes,
                self.exp.test_conf,
                self.exp.nmsthre
            )
        
        if outputs[0] is not None:
            online_targets = self.tracker.update(
                outputs[0].cpu().numpy(),
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            return online_targets
        return []
    
    def visualize(self, frame, online_targets):
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
        
        return plot_tracking(
            frame,
            online_tlwhs,
            online_ids,
            frame_id=0,
            fps=0.0
        )

def main():
    # Initialize tracker
    tracker = LiveStreamTracker(
        "bytetrack_m_mot17.pth.tar",
        "/home/cma/hucou/ByteTrack/exps/example/mot/yolox_m_mix_det.py"
    )
    
    # Open video stream
    video_path = "/home/cma/hucou/t6.mp4"
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        online_targets = tracker.detect_and_track(frame)
        result_frame = tracker.visualize(frame, online_targets)
        
        cv2.imshow('ByteTrack-Medium-MOT17', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()