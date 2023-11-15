import cv2
import torch
from torchvision.transforms.functional import normalize

from face_restoration_worker.utils.basicsr.archs.rrdbnet_arch import RRDBNet
from face_restoration_worker.utils.basicsr.utils import img2tensor, tensor2img
from face_restoration_worker.utils.basicsr.utils.realesrgan_utils import RealESRGANer
from face_restoration_worker.utils.basicsr.archs.codeformer_arch import CodeFormer

from face_restoration_worker.utils.facelib.utils.face_restoration_helper import (
    FaceRestoreHelper,
)
from face_restoration_worker.utils.facelib.utils.misc import is_gray


model_realesrgan = "/app/models/RealESRGAN_x2plus.pth"
model_codeformer = "/app/models/codeformer.pth"
model_detection = "retinaface_resnet50"
device = "cpu"


def set_realesrgan():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    upsampler = RealESRGANer(
        model_path=model_realesrgan,
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=False,
        scale=2,
    )
    return upsampler


def set_face_helper(upscale):
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=model_detection,
        save_ext="png",
        use_parse=True,
        device=device,
    )
    return face_helper


upsampler = set_realesrgan()


def set_codeformer_net():
    model = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    model.load_state_dict(torch.load(model_codeformer)["params_ema"])
    model.eval()
    return model


def set_face_helper(upscale):
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=model_detection,
        save_ext="png",
        use_parse=True,
        device=device,
    )
    return face_helper


def set_upscale(upscale, img):
    upscale = int(upscale)
    if upscale > 4:
        upscale = 4
    if upscale > 2 and max(img.shape[:2]) > 1000:
        upscale = 2
    if max(img.shape[:2]) > 1500 or upscale <= 0:
        upscale = 1
    return upscale


codeformer_net = set_codeformer_net()


def inference(
    image,
    background_enhance: bool = True,
    face_upsample: bool = True,
    upscale: int = 2,
    codeformer_fidelity: float = 0.5,
):
    has_aligned = False
    only_center_face = False
    draw_box = False

    # try:
    source = "blurry_face.jpg"
    img = cv2.imread(source, cv2.IMREAD_COLOR)

    upscale = set_upscale(upscale, img)
    if upscale == 1:
        background_enhance = False
        face_upsample = False

    face_helper = set_face_helper(upscale)
    bg_upsampler = upsampler if background_enhance else None
    face_upsampler = upsampler if face_upsample else None

    upscale = set_upscale(upscale, img)
    if upscale == 1:
        background_enhance = False
        face_upsample = False

    if has_aligned:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(img, threshold=5)
        if face_helper.is_gray:
            print("\tgrayscale input: True")
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        print(f"\tdetect {num_det_faces} faces")
        face_helper.align_warp_face()

    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = codeformer_net(
                    cropped_face_t, w=codeformer_fidelity, adain=True
                )[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f"Failed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    if not has_aligned:
        if bg_upsampler is not None:
            bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        if face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_box
            )

    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    return restored_img
    # except Exception as error:
    #     print("Global exception", error)
    #     return None, None
