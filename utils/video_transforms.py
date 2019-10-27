import torch
import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        """
        Args: 
            transforms (List[transform]): List of transform
        """
        self.transforms = transforms

    def __call__(self, video):
        for transform in self.transforms:
            video = transform(video)

        return video


class VideoCompose(Compose):
    pass


class VideoPadding:
    def __init__(self, seq_length):
        """
        Args:
            seq_length (int)
        """
        self.seq_length = seq_length
       
    
    def __call__(self, video):
        """
        Args:
            video (ndarray[C, T, H, W])
        Returns:
            video (ndarray[C, seq_length, H, W])
        """
        
        num1 = video.shape[1]
        num2 = video.shape[2]
        num3 = video.shape[3]
        
        padding_video = np.reshape([0]*num3*num2*3*(self.seq_length-num1),((self.seq_length-  num1),3,num2,num3)) 
        video = np.transpose(video,(1,0,2,3))
        video = np.r_[video,padding_video]
        video = np.transpose(video,(1,0,2,3)) 
        return video
    
class VideoResize:
    def __init__(self,size):
        self.size = size
        '''
        Args:
            size (tuple (H, W))
        '''
        
    
    def __call__(self, video):
        
        '''
        Args:
            video (ndarray[C, T, H, W])
        Returns:
            video (ndarray[C, T, H, W])
        '''
        
        output_area = self.size[0]*self.size[1]
        area = video.shape[2]*video.shape[3]
        if output_area < area:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        video = np.transpose(video,(1,2,3,0))
        
        clips = [np.expand_dims(cv2.resize(clip,self.size[::-1],interpolation = interpolation),axis = 0) for clip in video]
        video = np.concatenate(clips,axis = 0).transpose(3,0,1,2)
        #print('connected')
        
        return video
    

class VideoNormalization():
    def __init__(self):
        '''
        Args :
            Video Normalization
            
        Returns:
            Video (-1 ~ +1)
        '''
        pass
    
    def __call__(self, video):
        video = video.astype(np.float32)
        #print('1 video dtype',video.dtype, type(video))
        video -= 255.0
        video /= 127.5
        #print('2 video dtype', video.dtype)
        return video
    
    
    
    
def train_compose(seq_length,size):
    '''
    Args:
        seq_length (int)
        size (tuple (H, W))
    '''
    compose = VideoCompose([
        VideoResize(size),
        VideoPadding(seq_length),
        VideoNormalization(),
    ])
    return compose
