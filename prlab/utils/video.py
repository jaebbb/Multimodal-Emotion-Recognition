##########################################################
# Author : Tai Do
# Version: 0.1
# Version: 2018-06-08
##########################################################
import argparse, sys, cv2, time, os

class VideoReader(object):
    def __init__(self, source = 'camera', path = 0, extension='.jpg', extra = {}, **kwargs):
        self.params = {
            "source": source, "path": path, "extension": extension, 
            "FnInit"   : VideoReader.FnInit, 
            "FnProcess": VideoReader.FnProcess, 
            "FnShow"   : VideoReader.FnShow, 
            "FnIsExit" : VideoReader.FnIsExit, 
            "FnExit"   : VideoReader.FnExit,
            "sender"   : self,
        }
        self.params.update(extra)
        self.params.update(**kwargs)
        pass

    def process(self):
        process_video(extra = self.params)
    
    def OnInit(self, args):
        pass

    def OnProcess(self, args):
        pass

    def OnShow(self, args):
        pass

    def OnIsExit(self, args):
        if args['key'] == 27:
            args.update({"out_exit":True})
        pass

    def OnExit(self, args):
        pass

    @staticmethod
    def FnInit(sender, args):
        sender.OnInit(args)
        pass

    @staticmethod
    def FnProcess(sender, args):
        sender.OnProcess(args)
        pass

    @staticmethod
    def FnShow(sender, args):
        sender.OnShow(args)
        pass

    @staticmethod
    def FnIsExit(sender, args):
        sender.OnIsExit(args)
        pass

    @staticmethod
    def FnExit(sender, args):
        sender.OnExit(args)
        pass

# VideoReader

def track_info(**kwargs):
    '''
    Help for process_video function to output result
    '''
    params = {'output' : None, 'content' : 'hello', 'verbose' : 1, 'end_line' : '\n', 'mode' : 'at' }
    params.update(kwargs)
    if params['verbose'] == 1:
        if params['output'] is not None:
            writer = open(params["output"], params['mode'])
        else:
            writer = sys.stdout
        writer.write(params['content'])
        writer.write(params['end_line'])
        if params['output'] is not None:
            writer.close()
    # if
#def track_info

def process_video(source = 'camera', path = 0, extension='.jpg', verbose = 1, FnInit = None, FnProcess = None, FnShow = None, FnIsExit = None, FnExit = None, sender = None, extra = {}, **kwargs):
    '''
        process a video (camera, video, files) for process
        FnHandler(sender, args) --> sender: object call, args: parameters

        source    = ['video', 'camera', 'files'], video type
        path      = device_id, video or file path
        extension = file extension, .jpg
        delay     = 5
        show_info = None, {'fps', 'frame_idx', 'frame_cnt', 'total_time_used', 'delta_time'}
        show_prop = {'pos': (5, 20), 'font_face': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': 0.7, 'font_color': (0, 0, 255), 'thickness': 1}

        debug     = False
        FnProcess = Process Video
        show_win  = True
        title_win = Video
        show_scale= None, 0.5 or tuple

        input     = write video input
        output    = write video output
        log       = write log


        sender    = None
        extra     = "all"
        reader    = video/files/camera reader (need to get extra information)
    '''
    # Load parameters
    defaults = {'source' : source, 'path' : path, 'extension': extension, 'verbose' : verbose, 
              'delay' : 1, 'show_info': None, 'debug' : False, 'show_win' : True,  'title_win': 'Video',
              'FnInit':None, 'FnProcess': None, 'FnShow': None, 'FnIsExit':None, 'FnExit': None,
              'log': None, 'input': None, 'output': None, 'show_scale': None, "sender" : sender, "reader": None,
              'show_prop' :{'pos': (5, 20), 'font_face': cv2.FONT_HERSHEY_TRIPLEX, 'font_scale': 0.7, 'font_color': (0, 0, 255), 'thickness': 1}
    }
    params = {}
    params.update(defaults)
    params.update(kwargs)
    params.update(extra)
    
    for key in defaults['show_prop'].keys():
        if params['show_prop'].get(key) is None:
            params['show_prop'][key] = defaults['show_prop'][key]
    # for


    source = params["source"]
    path   = params["path"]
    extension = params["extension"]
    verbose = params["verbose"]    

    track_info(output = params['log'], content = "", verbose = verbose, end_line = '', mode = 'wt')

    # Load video or image sequence
    image = None
    flag  = False
    if params["source"] =='camera':
        video_reader = cv2.VideoCapture(int(path))
        flag, image = video_reader.read()
        name = "camera[%d]" % (path)
    elif params["source"]=='video':
        path         = os.path.abspath(path)
        video_reader = cv2.VideoCapture(path)
        flag, image = video_reader.read()
        name = "video[%s]" % (path)
    elif params["source"]=='files':
        files = [os.path.join(params["path"], x) for x in os.listdir(params["path"]) if x. endswith(params["extension"])]
        files.sort()
        name = "files[%s]" % (path)
        if len(files) > 0:
            image = cv2.imread(files[0])
            flag  = True

    if flag == False or image is None:
        return False

    # Load source information        
    height, width, channels = image.shape

    track_info(output = params['log'], content = "[info] Starting to read a sequence: %s" % (name), verbose = verbose)

    # Initialize creating input video  (for case loading from files)
    if params["input"] is not None:
        writerInput = cv2.VideoWriter(params["input"],  cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
    
    # Initialize creating output video  (for case writting result)
    if params["output"] is not None:
        writerOutput = cv2.VideoWriter(params["output"],  cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

    if params['FnInit'] is not None:
        params['FnInit'](sender = params["sender"], args = {'image': image, 'image_info': (height, width, channels)})

    # Init Processing Video    
    frame_idx = 0
    frame_cnt = 0
    total_time_used = 0
    args      = {}      # params for callback function
    if params["source"] == 'camera' or params["source"] == 'video':
        args['reader'] = video_reader
    else:
        args['reader'] = files

    # Start time
    start = time.time()
    # Loop Processing Video
    while(True):
        # READ A FRAME
        if params["source"] == 'camera' or params["source"] == 'video':
            flag, frame = video_reader.read()
        elif params["source"] == 'files':
            if frame_idx >= len(files):
                break
            frame = cv2.imread(files[frame_idx])
        if frame is None:
            break
        frame_idx = frame_idx + 1
        frame_cnt = frame_cnt + 1
        
        if params["source"] == 'video':
            args['timestamp'] = video_reader.get(cv2.CAP_PROP_POS_MSEC)

        # Output before processing
        if params["input"] is not None:
            writerInput.write(frame)
        
        # Benmarks
        delta_frame_time = time.time() - start    # time to process a frame
        if delta_frame_time>0:
            fps        = 1.0 / float(delta_frame_time) # frame per second 
        else:
            fps = 0       
        total_time_used += delta_frame_time        # total time
        start      = time.time()             # record time to start process

        track_info(output = params['log'], content = "[info] Processing Frame [%5d]" % (frame_idx), end_line ='', verbose = verbose)
        args.update({'frame': frame, 'frame_idx': frame_idx, 'fps': fps, 
                     'total_frame': frame_cnt, 'total_time': total_time_used, 'delta_frame_time':delta_frame_time})

        # PROCESS FRAME
        
        if params['FnProcess'] is not None:
            params['FnProcess'](sender = params["sender"], args = args)
        
        delta_process_time = time.time() - start    # time to process a frame
        args.update({'delta_process_time': delta_process_time})

        track_info(output = params['log'], content = "\tTime: %.5f (s)\tFPS: %d" %(delta_frame_time, fps), verbose = verbose)
        
        # Write FPS on Frame
        if params["show_info"] is not None:
            cv2.putText(frame, params["show_info"].format_map(args), 
                params["show_prop"]["pos"], params["show_prop"]["font_face"], params["show_prop"]["font_scale"], 
                params["show_prop"]["font_color"], params["show_prop"]["thickness"])

        if params['FnShow'] is not None:
            args.update({'frame': frame})
            params['FnShow'](sender = params["sender"], args = args)

        if params["show_win"] == True:
            if params["show_scale"] is not None:
                if params["show_scale"] is tuple:
                    frame1 = cv2.resize(frame, params["show_scale"])
                else:
                    frame1 = cv2.resize(frame, (int(width * params["show_scale"]), int(height * params["show_scale"])))
                cv2.imshow(params["title_win"], frame1)
            else:
                cv2.imshow(params["title_win"], frame)
        # show_win

        # Output after processing
        if params["output"] is not None:
            writerOutput.write(frame)

        # PROCESS NEXT FRAME
        key = -1
        if params['debug'] == True:
            key = cv2.waitKey(0)
        else:
            if params["source"] == 'camera' and params['delay'] == 0:
                key = cv2.waitKey(2)
            else:
                key = cv2.waitKey(params['delay'])
        
        args.update({'key': key, 'out_exit': False})
        if params['FnIsExit'] is not None:
            params['FnIsExit'](sender = params["sender"], args = args)
            if args['out_exit'] == True:
                break
        else:
            if key == 27:
                break
        cv2.waitKey(2)
    # end while
    args.update({'total_time':total_time_used, 'total_frame': frame_cnt, 'avg_fps': frame_cnt / total_time_used})
    if params['FnExit'] is not None:
        params['FnExit'](sender = params["sender"], args = args)
    track_info(output = params['log'], content = "[info] Stopping to read a sequence ...", verbose = verbose)
    track_info(output = params['log'], content = "Total time = %.5f (s)" % (total_time_used), verbose = verbose)
    track_info(output = params['log'], content = "FPS = %d" % (frame_cnt / total_time_used), verbose = verbose)
    track_info(output = params['log'], content = "Total frame = %d" % (frame_cnt), verbose = verbose)
    if params["show_win"] == True:
        cv2.destroyWindow(params["title_win"])
##  process_video ###

