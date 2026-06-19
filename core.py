import os
import shutil
import cv2
import sys
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class LogStream:
    def __init__(self, log_callback, check_cancel_callback):
        self.log_callback = log_callback
        self.check_cancel_callback = check_cancel_callback

    def write(self, text):
        cleaned_text = text.strip()
        if cleaned_text:
            if self.check_cancel_callback():
                raise InterruptedError("Process terminated by the user.")
            self.log_callback(cleaned_text)

    def flush(self):
        pass


class ImageProcessor:
    @staticmethod
    def analyze_and_enhance(input_path, save_path, scale_mode, progress_callback, log_callback, check_cancel_cb):
        current_scale = 2 if scale_mode == "2x" else 4
        log_callback(f"[Pipeline] Scaling factor set to: {scale_mode}")

        if not os.path.exists('temp'):
            os.makedirs('temp')

        temp_input = os.path.join('temp', 'safe_input.jpg')
        shutil.copy(input_path, temp_input)
        temp_output = os.path.join('temp', 'safe_output.jpg')

        img = cv2.imread(temp_input)

        if check_cancel_cb(): raise InterruptedError("Process terminated by the user.")

        progress_callback("Status: Analyzing input image properties...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        saturation_mean = s.mean()

        log_callback(f"[Analysis] Sharpness score: {laplacian_var:.2f}")
        log_callback(f"[Analysis] Saturation score: {saturation_mean:.2f}")

        if laplacian_var > 500:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            log_callback("[Process] -> Sharp/Noisy image. Mild blur applied to smooth out edges.")

        if saturation_mean > 120:
            blend_original_weight, blend_output_weight, clip_limit = 0.15, 0.85, 1.0
        elif saturation_mean < 40:
            blend_original_weight, blend_output_weight, clip_limit = 0.40, 0.60, 1.8
        else:
            blend_original_weight, blend_output_weight, clip_limit = 0.25, 0.75, 1.3

        if check_cancel_cb(): raise InterruptedError("Process terminated by the user.")

        height, width, _ = img.shape
        max_side = max(height, width)

        if current_scale == 4:
            if max_side <= 300:
                adaptive_tile = 128
            elif max_side <= 1000:
                adaptive_tile = 160
            else:
                adaptive_tile = 192
            log_callback(f"[Tiling] [4x Safety Mode] Selected tile size: {adaptive_tile}")
        else:
            if max_side <= 512:
                adaptive_tile = 0
            elif max_side <= 1200:
                adaptive_tile = 256
            else:
                adaptive_tile = 400
            log_callback(f"[Tiling] [2x Mode] Selected tile size: {adaptive_tile}")

        progress_callback("Status: AI processing on CPU... (Please wait)")

        model_filename = 'RealESRGAN_x2plus.pth' if current_scale == 2 else 'RealESRGAN_x4plus.pth'
        model_path = os.path.join('models', model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Required model file not found at: {model_path}")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=current_scale)

        upsampler = RealESRGANer(
            scale=current_scale, model_path=model_path, model=model,
            tile=adaptive_tile, tile_pad=10, pre_pad=0, half=False, gpu_id=None
        )

        if check_cancel_cb(): raise InterruptedError("Process terminated by the user.")

        log_callback(f"[AI Engine] Enhancing resolution ({scale_mode} scaling)...")

        old_stdout = sys.stdout
        sys.stdout = LogStream(log_callback, check_cancel_cb)

        try:
            output, _ = upsampler.enhance(img, outscale=current_scale)
        finally:
            sys.stdout = old_stdout

        if check_cancel_cb(): raise InterruptedError("Process terminated by the user.")

        progress_callback("Status: Adjusting color and contrast texture...")
        img_resized = cv2.resize(img, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        output = cv2.addWeighted(output, blend_output_weight, img_resized, blend_original_weight, 0)

        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        output = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        log_callback("[System] Saving final image and clearing temp files...")
        cv2.imwrite(temp_output, output)
        shutil.move(temp_output, save_path)
        if os.path.exists(temp_input):
            os.remove(temp_input)
        log_callback(f"[System] File saved successfully.")