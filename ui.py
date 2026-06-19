import os
import time
import threading
import shutil
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox
from core import ImageProcessor


class UpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Upscaler")
        self.geometry("640x720")
        self.resizable(True, True)
        self.minsize(580, 650)

        if os.path.exists("app_icon.ico"):
            self.iconbitmap("app_icon.ico")
        elif os.path.exists(os.path.join("_internal", "app_icon.ico")):
            self.iconbitmap(os.path.join("_internal", "app_icon.ico"))

        self.input_path = None
        self.raw_pil_image = None
        self.is_processing = False
        self.cancel_requested = False
        self.start_time = 0
        self.current_font_size = 12
        self.preview_base_size = 200  # Default scale dimension for thumbnail

        self.setup_ui()

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        self.tab_view.add("Dashboard")
        self.tab_view.add("Logs")
        self.tab_view.add("Help")

        # ----------------------------------------------------
        # TAB 1: DASHBOARD
        # ----------------------------------------------------
        dash_tab = self.tab_view.tab("Dashboard")
        dash_tab.grid_rowconfigure(0, weight=1)
        dash_tab.grid_columnconfigure(0, weight=1)

        self.scrollable_frame = ctk.CTkScrollableFrame(dash_tab, fg_color="transparent")
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Title Section
        self.title_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="Real-ESRGAN Super Resolution Engine - Made By: @Arlekcino007",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.pack(pady=(10, 15))

        # CARD 1: File Selection Card
        self.action_card = ctk.CTkFrame(self.scrollable_frame, border_width=1, border_color="#3D3D3D")
        self.action_card.pack(fill="x", padx=15, pady=6)

        self.select_btn = ctk.CTkButton(self.action_card, text="Select Input Image", width=180,
                                        command=self.select_image)
        self.select_btn.pack(pady=(15, 8))

        self.path_label = ctk.CTkLabel(self.action_card, text="No file selected", text_color="gray",
                                       font=ctk.CTkFont(size=12))
        self.path_label.pack(pady=(0, 15))

        # CARD 2: Interactive Preview Card
        self.preview_card = ctk.CTkFrame(self.scrollable_frame, border_width=1, border_color="#3D3D3D")
        self.preview_card.pack(fill="x", padx=15, pady=6)

        preview_ctrl_bar = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_ctrl_bar.pack(fill="x", padx=15, pady=(8, 0))

        preview_title = ctk.CTkLabel(preview_ctrl_bar, text="Image Monitor Box",
                                     font=ctk.CTkFont(size=12, weight="bold"))
        preview_title.pack(side="left")

        self.btn_zoom_out_img = ctk.CTkButton(preview_ctrl_bar, text="-", width=28, height=22, state="disabled",
                                              command=lambda: self.resize_preview(-30))
        self.btn_zoom_out_img.pack(side="right", padx=2)

        self.btn_zoom_in_img = ctk.CTkButton(preview_ctrl_bar, text="+", width=28, height=22, state="disabled",
                                             command=lambda: self.resize_preview(30))
        self.btn_zoom_in_img.pack(side="right", padx=2)

        self.preview_frame = ctk.CTkFrame(self.preview_card, fg_color="#222222", height=240)
        self.preview_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="No image to display", text_color="gray")
        self.preview_label.pack(expand=True, pady=15)

        # CARD 3: Technical Configurations Card
        self.settings_card = ctk.CTkFrame(self.scrollable_frame, border_width=1, border_color="#3D3D3D")
        self.settings_card.pack(fill="x", padx=15, pady=6)

        self.scale_label = ctk.CTkLabel(self.settings_card, text="Select AI Model Scale Factor:",
                                        font=ctk.CTkFont(size=12, weight="bold"))
        self.scale_label.pack(pady=(12, 6))

        self.scale_var = ctk.StringVar(value="2x")
        self.scale_selector = ctk.CTkSegmentedButton(
            self.settings_card,
            values=["2x", "4x"],
            variable=self.scale_var,
            command=self.on_scale_changed
        )
        self.scale_selector.pack(pady=(0, 15))

        # Metrics Status Monitors
        self.stats_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=15, pady=8)

        self.progress_label = ctk.CTkLabel(self.stats_frame, text="Status: Ready to work",
                                           font=ctk.CTkFont(size=13, weight="bold"))
        self.progress_label.pack(pady=2)

        self.timer_label = ctk.CTkLabel(self.stats_frame, text="Elapsed Time: 00:00",
                                        font=ctk.CTkFont(family="Courier", size=13), text_color="#A0A0A0")
        self.timer_label.pack(pady=2)

        # Operational Execution Anchors
        self.controls_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.controls_frame.pack(pady=10)

        self.start_btn = ctk.CTkButton(
            self.controls_frame, text="Enhance & Save", command=self.start_processing_thread,
            state="disabled", fg_color="green", width=140
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ctk.CTkButton(
            self.controls_frame, text="Stop Process", command=self.request_cancel,
            state="disabled", fg_color="#D32F2F", hover_color="#B71C1C", width=140
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        # ----------------------------------------------------
        # TAB 2: LOGS
        # ----------------------------------------------------
        log_tab = self.tab_view.tab("Logs")
        log_tab.grid_rowconfigure(1, weight=1)
        log_tab.grid_columnconfigure(0, weight=1)

        zoom_frame = ctk.CTkFrame(log_tab, fg_color="transparent")
        zoom_frame.grid(row=0, column=0, sticky="ne", padx=10, pady=(5, 0))

        zoom_title = ctk.CTkLabel(zoom_frame, text="Console Zoom: ", font=ctk.CTkFont(size=11))
        zoom_title.pack(side="left", padx=2)

        btn_zoom_in = ctk.CTkButton(zoom_frame, text="+", width=30, height=25, command=self.zoom_in_logs)
        btn_zoom_in.pack(side="left", padx=2)

        btn_zoom_out = ctk.CTkButton(zoom_frame, text="-", width=30, height=25, command=self.zoom_out_logs)
        btn_zoom_out.pack(side="left", padx=2)

        self.log_box = ctk.CTkTextbox(log_tab, font=ctk.CTkFont(family="Courier", size=self.current_font_size))
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.log_box.configure(state="disabled")

        # ----------------------------------------------------
        # TAB 3: HELP & USAGE
        # ----------------------------------------------------
        help_tab = self.tab_view.tab("Help")
        help_tab.grid_rowconfigure(0, weight=1)
        help_tab.grid_columnconfigure(0, weight=1)

        help_scroller = ctk.CTkScrollableFrame(help_tab, fg_color="transparent")
        help_scroller.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        help_text = (
            "📖 AI Image Upscaler User Guide\n"
            "--------------------------------------------------\n\n"
            "1. How to Use the Program:\n"
            "   • Click 'Select Input Image' on the Dashboard to load an image.\n"
            "   • View its native resolution metrics dynamically displayed next to the filename.\n"
            "   • Use the '+' and '-' monitors to resize the preview workspace without affecting the output resolution.\n"
            "   • Choose a Scale Factor (2x or 4x scale metric metrics).\n"
            "   • Click 'Enhance & Save', choose your destination path, and wait for the AI process to complete.\n\n"
            "2. Under the Hood (How it Works):\n"
            "   • Adaptive Analysis: Analyzes Laplacian variance for sharpness and HSV split matrices for saturation.\n"
            "   • Automated Blur Control: Smooths highly pixelated edges dynamically.\n"
            "   • Smart Tiling Engine: Processes the image in safety-bounded blocks to prevent system memory overflows.\n"
            "   • Advanced Post-Processing: Blends raw structures using Lanczos4 matrices and sharpens textures via CLAHE execution."
        )

        help_label = ctk.CTkLabel(
            help_scroller,
            text=help_text,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12, family="Segoe UI")
        )
        help_label.pack(fill="both", expand=True, padx=10, pady=5)

    def zoom_in_logs(self):
        if self.current_font_size < 24:
            self.current_font_size += 1
            self.log_box.configure(font=ctk.CTkFont(family="Courier", size=self.current_font_size))

    def zoom_out_logs(self):
        if self.current_font_size > 8:
            self.current_font_size -= 1
            self.log_box.configure(font=ctk.CTkFont(family="Courier", size=self.current_font_size))

    def on_scale_changed(self, value):
        if value == "4x":
            self.add_log("[UI] User switched to 4x Model. High-safety tiling metrics will be applied.")
        else:
            self.add_log("[UI] User switched to 2x Model. Balanced performance mode activated.")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if file_path:
            self.input_path = file_path
            self.raw_pil_image = Image.open(file_path)
            orig_w, orig_h = self.raw_pil_image.size

            display_text = f"File: {os.path.basename(file_path)}  |  Resolution: {orig_w}x{orig_h} px"
            self.path_label.configure(text=display_text, text_color="white")

            self.btn_zoom_in_img.configure(state="normal")
            self.btn_zoom_out_img.configure(state="normal")
            self.preview_base_size = 200

            self.render_preview_image()

            if not self.is_processing:
                self.start_btn.configure(state="normal")
            self.add_log(f"[UI] Input image loaded: {os.path.basename(file_path)} [{orig_w}x{orig_h}]")

    def render_preview_image(self):
        if self.raw_pil_image:
            img_copy = self.raw_pil_image.copy()
            img_copy.thumbnail((self.preview_base_size, self.preview_base_size))

            ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)
            self.preview_label.configure(image=ctk_img, text="")

    def resize_preview(self, scale_delta):
        new_size = self.preview_base_size + scale_delta
        if 100 <= new_size <= 450:
            self.preview_base_size = new_size
            self.render_preview_image()

    def update_progress(self, text):
        self.progress_label.configure(text=text)

    def add_log(self, message):
        self.after(0, self._execute_add_log, message)

    def _execute_add_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def update_timer_loop(self):
        if self.is_processing:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_label.configure(text=f"Elapsed Time: {mins:02d}:{secs:02d}")
            self.after(1000, self.update_timer_loop)

    def request_cancel(self):
        if self.is_processing:
            self.cancel_requested = True
            self.update_progress("Status: Canceling operation...")
            self.add_log("[UI] Termination request sent. Cleaning memory pipelines...")
            self.stop_btn.configure(state="disabled")

    def check_cancel_flag(self):
        return self.cancel_requested

    def start_processing_thread(self):
        threading.Thread(target=self.process_image, daemon=True).start()

    def process_image(self):
        try:
            self.is_processing = True
            self.cancel_requested = False

            self.start_btn.configure(state="disabled")
            self.select_btn.configure(state="disabled")
            self.scale_selector.configure(state="disabled")
            self.stop_btn.configure(state="normal")

            selected_scale = self.scale_var.get()

            save_path = filedialog.asksaveasfilename(
                title="Select Output Destination",
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                initialfile=f"enhanced_image_{selected_scale}.jpg"
            )
            if not save_path:
                self.update_progress("Status: Operation canceled")
                self.add_log("[UI] Enhancement operation canceled by the user.")
                return

            self.add_log(f"[UI] Launching pipeline for {selected_scale} scale factor...")

            self.start_time = time.time()
            self.after(0, self.update_timer_loop)

            ImageProcessor.analyze_and_enhance(
                self.input_path,
                save_path,
                selected_scale,
                self.update_progress,
                self.add_log,
                self.check_cancel_flag
            )

            total_elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(total_elapsed, 60)

            self.update_progress("Status: Process completed successfully!")
            self.add_log(f"[UI] Process completed successfully in {mins:02d}:{secs:02d}")
            messagebox.showinfo("Success",
                                f"Image enhanced to {selected_scale} successfully!\nTotal Time: {mins:02d}:{secs:02d}")

        except InterruptedError:
            self.update_progress("Status: Process Stopped")
            self.add_log("[UI] Pipeline completely stopped. Temporary folders recycled.")
            messagebox.showwarning("Stopped", "The upscaling operation was safely stopped by the user.")

            if os.path.exists('temp'):
                shutil.rmtree('temp')

        except Exception as e:
            self.update_progress("Status: Processing error occurred")
            self.add_log(f"[CRITICAL ERROR] {str(e)}")
            messagebox.showerror("System Error", f"An unexpected error occurred:\n{str(e)}")
        finally:
            self.is_processing = False
            self.start_btn.configure(state="normal")
            self.select_btn.configure(state="normal")
            self.scale_selector.configure(state="normal")
            self.stop_btn.configure(state="disabled")