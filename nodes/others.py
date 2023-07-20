from .util import common_annotator_call, img_np_to_tensor
from ..v11 import tile, inpaint, inpaint_lama
from .. import mp_face_mesh, color

class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}), #Which image has more than 50 detectable faces?
                              "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.1})
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/face_mesh"

    def detect(self, image, max_faces, min_confidence):
        np_detected_map = common_annotator_call(mp_face_mesh.generate_annotation, image, max_faces, min_confidence)
        return (img_np_to_tensor(np_detected_map),)



class Color_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",) }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_processed_pallete"

    CATEGORY = "preprocessors/color_style"

    def get_processed_pallete(self, image):
        np_detected_map = common_annotator_call(color.apply_color, image)
        return (img_np_to_tensor(np_detected_map),)



class Tile_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "pyrUp_iters": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}) }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "preprocessors/tile"

    def preprocess(self, image, pyrUp_iters):
        np_detected_map = common_annotator_call(tile.preprocess, image, pyrUp_iters)
        return (img_np_to_tensor(np_detected_map),)
    
class Inpaint_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "preprocessors/inpaint"

    def preprocess(self, image, mask):
        return (inpaint.preprocess(image, mask),)
    
class Inpaint_Lama_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "preprocessors/inpaint"
    def preprocess(self, image, mask):
        return inpaint_lama.preprocess(image, mask)

NODE_CLASS_MAPPINGS = {
    "MediaPipe-FaceMeshPreprocessor": Media_Pipe_Face_Mesh_Preprocessor,
    "ColorPreprocessor": Color_Preprocessor,
    "TilePreprocessor": Tile_Preprocessor,
    "InpaintPreprocessor": Inpaint_Preprocessor,
    "InpaintLamaPreprocessor": Inpaint_Lama_Preprocessor
}
