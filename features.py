import os
import cv2
import h5py
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def load_models(device):

    yolo_det = YOLO('../yolov12s.pt')
    yolo_det.to(device)
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 8

    session = ort.InferenceSession(
        '../depth_anything_v2_vitb.onnx', 
        sess_options=sess_options, 
        providers=['CUDAExecutionProvider'],
    )
    return yolo_det, session

def frame_indices(total_frames, count=16):
    linear = np.linspace(0, 1, count)
    eased = linear ** 0.5   ##
    indices = (eased * (total_frames - 1)).astype(int)
    return indices

def compute_optical_flow(frame1, frame2):
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1], angleInDegrees=True)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    ang = cv2.normalize(ang, None, 0, 1, cv2.NORM_MINMAX)
    flow_processed = np.stack([mag, ang], axis=-1)
    return flow_processed


def save_video_features(video_features, video_id, output_dir="../test_video_features"):
    file_path = os.path.join(output_dir, f"{video_id}.h5")
    frames, mask_features, depth_channel, optical_flow = video_features

    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset('frames', data=frames, compression="gzip", compression_opts=1)
        h5f.create_dataset('mask_features', data=mask_features, compression="gzip", compression_opts=1)
        h5f.create_dataset('depth_channel', data=depth_channel, compression="gzip", compression_opts=1)
        h5f.create_dataset('optical_flow', data=optical_flow, compression="gzip", compression_opts=1)
        h5f.attrs['video_id'] = video_id
    return file_path

class FeatureExtract:
    def __init__(self, path,df,alert_frame,depth_model,device,frame_size=(256,256),frame_numbers=24,transform=False):
        self.path = path
        self.df = df
        self.alert_frame = alert_frame
        # self.yolo_det = yolo_det
        self.depth_model = depth_model
        self.device = device
        self.frame_size = frame_size
        self.frame_numbers = frame_numbers
        self.transform = transform
        
        # Precompute constants
        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])

    def getitem(self, idx):

        time_of_alert = int(self.df.loc[idx,self.alert_frame])
        target = int(self.df.loc[idx,'target'])
        video_id = self.df.loc[idx, 'id']
        video_path = os.path.join(self.path, f"{video_id:05d}.mp4")
        
        cap = cv2.VideoCapture(video_path)
        frame_tobe_consider = sorted(frame_indices(time_of_alert-1, count=self.frame_numbers-1).tolist() + [time_of_alert])
        
        frames = []
        mask_features = []
        depth_channel = []
        # optical_flow = []
        
        # Create binding once
        binding = self.depth_model.io_binding()
        ort_input = self.depth_model.get_inputs()[0].name
        ort_output = self.depth_model.get_outputs()[0].name
        
        # Batch processing setup
        # raw_frames = []

        for frame_number in frame_tobe_consider:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f'Zeroes trigger for :- {video_id}')
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            img = cv2.cvtColor(frame[:720-80,:], cv2.COLOR_BGR2RGB)
            # raw_frames.append(img)
            img_resized = cv2.resize(img, self.frame_size)
            frames.append(img_resized)

        cap.release()

        # prev_frame = None
        # Now process all frames
        for i, frame in enumerate(frames):
            # Compute optical flow
            # if prev_frame is None:
            #     optical_flow.append(compute_optical_flow(frame, frame))
            # else:
            #     optical_flow.append(compute_optical_flow(prev_frame, frame))
            # prev_frame = frame
            
            # Prepare depth input 
            depth_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            depth_input = cv2.resize(depth_input, (518, 518), interpolation=cv2.INTER_CUBIC)
            depth_input = (depth_input - self.img_mean) / self.img_std
            depth_input = depth_input.transpose(2, 0, 1)[None].astype("float32")
            
            # Run depth model
            binding.bind_cpu_input(ort_input, depth_input)
            binding.bind_output(ort_output, self.device)
            self.depth_model.run_with_iobinding(binding)
            depth = binding.get_outputs()[0].numpy()
            
            # Normalize and resize depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.transpose(1, 2, 0).astype("uint8")
            depth = cv2.resize(depth, self.frame_size, interpolation=cv2.INTER_CUBIC)
            # depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
            depth_channel.append(depth/255.0)

        # results = self.yolo_det(raw_frames, verbose=False)
        
        # # Process YOLO results and create masks
        # for i, result in enumerate(results):
        #     class_ids = result.boxes.cls
        #     xywh = result.boxes.xywh
        #     filtered_boxes = xywh[(class_ids == 2) | (class_ids == 7)]
            
        #     combined_mask = np.zeros((640, 1280))
        #     for car in filtered_boxes.cpu().numpy():
        #         x_center, y_center, width, height = car
        #         x1 = int(x_center - width / 2)
        #         y1 = int(y_center - height / 2)
        #         x2 = int(x_center + width / 2)
        #         y2 = int(y_center + height / 2)
        #         combined_mask = cv2.rectangle(combined_mask, (x1, y1), (x2, y2), color=255, thickness=-1)
        #     combined_mask = cv2.resize(combined_mask , self.frame_size)
        #     mask_features.append(combined_mask/255.0)
            
        return (np.stack(frames)/255.0), np.stack(depth_channel), int(video_id) , int(target)
