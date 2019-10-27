from .tiny_face_all import TinyFaceDetection as TinyFace

class TinyFaceDetection(TinyFace):
    def get_info(self):
        return dict(name="tinyface", source="rgb", bbox_color=(128,128,0), object = self)
    pass
# TinyFaceDetection
