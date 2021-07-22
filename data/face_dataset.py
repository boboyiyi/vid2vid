import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2
from skimage import feature

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import interpPoints, drawEdge

class FaceDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot                
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_keypoints')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_img')
        
        # A_paths和B_paths是二维list，结构是[[视频1的帧], [视频2的帧], ..., [最后一个视频的帧]]，帧内是排序好的
        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        check_path_valid(self.A_paths, self.B_paths)

        self.init_frame_idx(self.A_paths)
        self.scale_ratio = np.array([[0.9, 1], [1, 1], [0.9, 1], [1, 1.1], [0.9, 0.9], [0.9, 0.9]])#np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_ratio_sym = np.array([[1, 1], [0.9, 1], [1, 1], [0.9, 1], [1, 1], [1, 1]]) #np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_shift = np.zeros((6, 2)) #np.random.uniform(-5, 5, size=[6, 2])

    def __getitem__(self, index):
        # 人脸任务中，该行只完成index % nb_seqs操作，A，B，I都是None
        # edge2face任务中batch_size = 1，所以这里
        A, B, I, seq_idx = self.update_frame_idx(self.A_paths, index)        
        A_paths = self.A_paths[seq_idx]
        B_paths = self.B_paths[seq_idx]
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), self.frame_idx)
        
        B_img = Image.open(B_paths[start_idx]).convert('RGB')
        B_size = B_img.size
        points = np.loadtxt(A_paths[start_idx], delimiter=',')
        # 是否为第一帧，这个应该是在infer阶段用的，Train阶段永远是True
        is_first_frame = self.opt.isTrain or not hasattr(self, 'min_x')
        if is_first_frame: # crop only the face region
            # train的时候只关注人脸区域，这里怎么得到的roi，很迷糊…
            self.get_crop_coords(points, B_size)
        # 按刚才得到的ROI crop B_img，get_img_params得到如下dict
        # {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}
        # edge2face任务中，new_w和new_h都能被32整除
        # crop_size为(0, 0)，crop_pos为(0, 0)，flip有50%的概率为True
        params = get_img_params(self.opt, self.crop(B_img).size)
        # edge2face任务中，transform_scaleA和transform_label都是resize到param指定的new_size ==> 50%概率水平flip==>ToTensor
        # transform_scaleB也是resize ==> 50% flip ==> ToTensor ==> Normalize
        transform_scaleA = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        # 这里对label插值的时候肯定不能用双线性，因为label一定是唯一的（不是眉毛就是眼睛）
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_scaleB = get_transform(self.opt, params)
        
        # read in images
        frame_range = list(range(n_frames_total)) if self.A is None else [self.opt.n_frames_G-1]
        for i in frame_range:
            # 这里就明确表示了t_step的作用，就是相邻帧可以隔帧取，但edge2face任务中t_step=1，所以还是连续的帧当成相邻帧
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]
            B_img = Image.open(B_path)
            # Ai：第i个输入，Li：第i个label
            Ai, Li = self.get_face_image(A_path, transform_scaleA, transform_label, B_size, B_img)
            Bi = transform_scaleB(self.crop(B_img))
            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)
            I = concat_frame(I, Li, n_frames_total)
        
        if not self.opt.isTrain:
            self.A, self.B, self.I = A, B, I
            self.frame_idx += 1
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'A': A, 'B': B, 'inst': I, 'A_path': A_path, 'change_seq': change_seq}
                
        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)                
        A_scaled = transform_scaleA(self.crop(A_img))
        return A_scaled

    def get_face_image(self, A_path, transform_A, transform_L, size, img):
        # read face keypoints from path and crop face region
        keypoints, part_list, part_labels = self.read_keypoints(A_path, size)

        # draw edges and possibly add distance transform maps
        add_dist_map = not self.opt.no_dist_map
        im_edges, dist_tensor = self.draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map)
        
        # canny edge for background
        # 对背景采用canny edge detection
        if not self.opt.no_canny_edge:
            edges = feature.canny(np.array(img.convert('L')))
            # (part_labels == 0)，非人脸区域都是0，乘以edges相当于去掉了人脸部分的edges
            edges = edges * (part_labels == 0)  # remove edges within face
            im_edges += (edges * 255).astype(np.uint8)
        edge_tensor = transform_A(Image.fromarray(self.crop(im_edges)))

        # final input tensor
        input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
        # input_tensor shape为[15, 512, 512]
        label_tensor = transform_L(Image.fromarray(self.crop(part_labels.astype(np.uint8)))) * 255.0
        return input_tensor, label_tensor

    def read_keypoints(self, A_path, size):
        # mapping from keypoints to face part 
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                     [range(60, 65), [64,65,66,67,60]]                 # tongue
                    ]
        label_list = [1, 2, 2, 3, 4, 4, 5, 6] # labeling for different facial parts
        # debug让part的颜色更重一些
        # label_list = [i * 30 for i in label_list]
        keypoints = np.loadtxt(A_path, delimiter=',')
        
        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        # 第1个点和第17个点的y中值
        baseline_y = (pts[0,1] + pts[-1,1]) / 2
        # 第2到16个点作为upper_pts
        upper_pts = pts[1:-1,:].copy()
        # 
        upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
        # upper_pts[::-1,:]将index翻转，比如之前是2 - 16，现在变成16 - 2
        # vstack之后的点数就是68 + 15 = 83点，这15点是把轮廓点做了个上下镜像（不完全镜像，有缩放因子2/3，上面的点更扁一些）
        keypoints = np.vstack((keypoints, upper_pts[::-1,:]))

        # label map for facial part 
        w, h = size
        part_labels = np.zeros((h, w), np.uint8)
        # 学会这种用法，同时得到index和内容！
        for p, edge_list in enumerate(part_list):
            # 将每一个part的点的集合整成一个向量，比如[[28, 31], range(31, 36), [35, 28]]，通过以下语句，indices输出
            # [28, 31, 31, 32, 33, 34, 35, 35, 28]
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            # 填充多边形，不同的part绘制不同的颜色
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])
        # cv2.imwrite('face_part.jpg', part_labels)

        # move the keypoints a bit
        if not self.opt.isTrain and self.opt.random_scale_points:
            self.scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
            self.scale_points(keypoints, part_list[4] + part_list[5], 3, sym=True)
            for i, part in enumerate(part_list):
                self.scale_points(keypoints, part, label_list[i]-1)

        return keypoints, part_list, part_labels

    def draw_face_edges(self, keypoints, part_list, transform_A, size, add_dist_map):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        dist_tensor = 0
        e = 1                
        for edge_list in part_list:
            for edge in edge_list:
                im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y)
                    if add_dist_map:
                        drawEdge(im_edge, curve_x, curve_y)
                                
                if add_dist_map: # add distance transform map on each facial part
                    # Distance Transform，计算非零点距离最近零点的距离，L1表示计算方法为|x2 - x1| + |y2 - y1|
                    # mask_size = 3
                    # 这里255 - im_edge，所以就只有edge的点是0，那么im_dist边缘的值就是最大的
                    im_dist = cv2.distanceTransform(255-im_edge, cv2.DIST_L1, 3)
                    # 给im_dist除以3，如果值还大于255，直接取255
                    # edge以外的部分是渐变的，边缘接近或等于255(亮度越高)，越往内越靠近0（越暗）
                    im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
                    im_dist = Image.fromarray(im_dist)
                    tensor_cropped = transform_A(self.crop(im_dist))                    
                    dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
                    e += 1
        # 这里dist_tensor的shape是[14, 512, 512]，parts中一共有14个region，上面的dist_map是cat在一起的
        return im_edges, dist_tensor

    def get_crop_coords(self, keypoints, size):
        # 得到人脸关键点的bounding box
        min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
        min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()
        # 
        xc = (min_x + max_x) // 2
        yc = (min_y*3 + max_y) // 4
        # 给bounding box的w乘以2.5，得到一个的正方形
        h = w = (max_x - min_x) * 2.5
        # 如果xc - w // 2小于0，则必然w大于size[0]，相当于xc = min(w, size[0]) - w // 2
        # 如果xc - w // 2大于0，则必然w小于size[0]，相当于xc = min(xc - w // 2 + w, size[0]) - w // 2
        # 
        xc = min(max(0, xc - w//2) + w, size[0]) - w//2
        yc = min(max(0, yc - h//2) + h, size[1]) - h//2
        # 这里因为h和w完全一致，所以最终的bounding box一定是一个正方形
        min_x, max_x = xc - w//2, xc + w//2
        min_y, max_y = yc - h//2, yc + h//2
        self.min_y, self.max_y, self.min_x, self.max_x = int(min_y), int(max_y), int(min_x), int(max_x)

    def crop(self, img):
        if isinstance(img, np.ndarray):
            return img[self.min_y:self.max_y, self.min_x:self.max_x]
        else:
            return img.crop((self.min_x, self.min_y, self.max_x, self.max_y))

    def scale_points(self, keypoints, part, index, sym=False):
        if sym:
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio_sym[index, 0]
            ratio_y = self.scale_ratio_sym[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]
            for idx in part:
                pts_i = keypoints[idx]
                mean_i = np.mean(pts_i, axis=0)
                mean_ix, mean_iy = mean_i[0], mean_i[1]
                new_mean_ix = (mean_ix - mean_x) * ratio_x + mean_x
                new_mean_iy = (mean_iy - mean_y) * ratio_y + mean_y
                pts_i[:,0] = (pts_i[:,0] - mean_ix) + new_mean_ix
                pts_i[:,1] = (pts_i[:,1] - mean_iy) + new_mean_iy
                keypoints[idx] = pts_i

        else:            
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio[index, 0]
            ratio_y = self.scale_ratio[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]            
            pts[:,0] = (pts[:,0] - mean_x) * ratio_x + mean_x + self.scale_shift[index, 0]
            pts[:,1] = (pts[:,1] - mean_y) * ratio_y + mean_y + self.scale_shift[index, 1]
            keypoints[pts_idx] = pts

    def __len__(self):
        if self.opt.isTrain:
            return len(self.A_paths)
        else:
            return sum(self.frames_count)

    def name(self):
        return 'FaceDataset'
