import os
import tkinter as tk
import numpy as np
import math
from PIL import Image, ImageTk, ImageDraw

DIRS = [
    "./../processed/Sony/long",
    "./../processed/Sony/short_linear_enhance",
    "./../processed/Sony/sid_original",
    "./../processed/Sony/sid_no_bottleneck"
]
LABELS = [
    "Long exposure",
    "Short exposure (linear enhanced)",
    "Learning to See in the Dark (2018)",
    "Learning to See in the Dark (no bottleneck)",
]

GROUND_TRUTH_INDEX = 0
CALCULATE_COMPARISON = [
    False, 
    True,
    True,
    True
]

FILE_PREFIX_LEN = 8
IMAGE_GAP = 10
TEXT_BAR_HEIGHT = 20

def list_pngs(folder):
    if not folder or not os.path.isdir(folder):
        return {}
    return {f[:FILE_PREFIX_LEN]: f for f in os.listdir(folder) if f.lower().endswith(".png")}

def load_scenes():
    pngs_per_dir = [list_pngs(d) for d in DIRS]
    common = set()
    for pngs in pngs_per_dir:
        common = common | set(pngs.keys())

    scenes = []
    for prefix in sorted(common):
        paths = []
        for i in range(4):
            if DIRS[i] and prefix in pngs_per_dir[i]:
                paths.append(os.path.join(DIRS[i], pngs_per_dir[i][prefix]))
            else:
                paths.append(None)
        scenes.append(paths)
    return scenes

class ImageView:
    def __init__(self, path):
        self.path = path
        self.image = Image.open(path).convert("RGB")
        self.comparison_text = None
    
    def calculate_comparison(self, ground_truth):
        im_array = np.array(self.image)
        gt_array = np.array(ground_truth)
        mse = ((im_array - gt_array)**2).mean()
        psnr = 20.0 * math.log10(255) - 10.0 * math.log10(mse)
        self.comparison_text = f"PSNR: {psnr:.2f}dB"

class QuadViewer:
    def __init__(self, root, scenes):
        self.root = root
        self.show_after_resize_callback = None
        
        self.scenes = scenes
        self.scene_index = 0
        self.image_views = [None, None, None, None]

        self.dragging = False
        self.zoom = 1.0
        self.crop_center = (0.5, 0.5)
        self.start_drag_crop_center = (0, 0)
        self.start_drag_mouse = (0, 0)

        root.geometry("1200x850")

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
               
        self.canvas_image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.tk_img = None

        root.bind("<Right>", self.next)
        root.bind("<Left>", self.prev)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<MouseWheel>", self.on_zoom)        # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)          # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)          # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.do_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)

        self.load_images()
        self.show()
        
    def after_resize(self):
        self.show_after_resize_callback = None
        self.canvas_image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.show()

    def on_resize(self, _event):
        if self.show_after_resize_callback is not None:
            self.root.after_cancel(self.show_after_resize_callback)
        self.show_after_resize_callback = self.root.after(100, self.after_resize)

    def current_canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        return w, h
    
    def current_cell_size(self):
        canvas_w, canvas_h = self.current_canvas_size()
        w = (canvas_w - IMAGE_GAP) // 2
        h = (canvas_h - IMAGE_GAP - 2 * TEXT_BAR_HEIGHT) // 2
        return w, h

    def load_images(self):
        self.image_views = [ImageView(path) if path else None for path in self.scenes[self.scene_index]]
        if self.image_views[GROUND_TRUTH_INDEX]:
            for i in range(0, 4):
                if CALCULATE_COMPARISON[i]:
                    self.image_views[i].calculate_comparison(self.image_views[GROUND_TRUTH_INDEX].image)
        
        self.zoom = 1.0
        self.crop_center = (0.5, 0.5)
        self.dragging = False
        
    def next(self, _event=None):
        self.scene_index = (self.scene_index + 1) % len(self.scenes)
        self.load_images()
        self.show()
            
    def prev(self, _event=None):
        self.scene_index = (self.scene_index + len(self.scenes) - 1) % len(self.scenes)
        self.load_images()
        self.show()

    def on_zoom(self, event):
        delta = 0
        if event.num == 4: 
            delta = 1
        elif event.num == 5:
            delta = -1
        elif event.delta:
            delta = 1 if event.delta > 0 else -1

        factor = 1.1 if delta > 0 else 0.9
        self.zoom = max(1.0, min(self.zoom * factor, 50.0))  # clamp
        
        half_crop_size = 0.5 / self.zoom
        cx, cy = self.crop_center
        cx = min(max(cx, half_crop_size), 1.0 - half_crop_size)
        cy = min(max(cy, half_crop_size), 1.0 - half_crop_size)
        self.crop_center = (cx, cy)
        
        self.show()
    
    def start_drag(self, event):
        self.dragging = True
        self.start_drag_mouse = (event.x, event.y)
        self.start_drag_crop_center = self.crop_center
        
    def stop_drag(self, event):
        self.dragging = False
        self.show()

    def do_drag(self, event):
        if not self.dragging:
            return
        
        dx = event.x - self.start_drag_mouse[0]
        dy = event.y - self.start_drag_mouse[1]

        cell_w, cell_h = self.current_cell_size()
        
        cx, cy = self.start_drag_crop_center
        cx -= dx / (cell_w * self.zoom)
        cy -= dy / (cell_h * self.zoom)

        half_crop_size = 0.5 / self.zoom

        cx = min(max(cx, half_crop_size), 1.0 - half_crop_size)
        cy = min(max(cy, half_crop_size), 1.0 - half_crop_size)
        
        self.crop_center = (cx, cy)
        self.show(fast_resample=True)

    def show(self, fast_resample = False):
        cell_w, cell_h = self.current_cell_size()
        if cell_w <= 0 or cell_h <= 0:
            return
        
        draw = ImageDraw.Draw(self.canvas_image)
        
        for i, view in enumerate(self.image_views):
            row, col = divmod(i, 2)
            x = col * (cell_w + IMAGE_GAP)
            y = row * (cell_h + IMAGE_GAP + TEXT_BAR_HEIGHT)

            if view:
                
                # Compute crop
                viewport_w = view.image.width / self.zoom
                viewport_h = view.image.height / self.zoom
                cx, cy = self.crop_center
                x1 = int(cx * view.image.width - viewport_w / 2)
                y1 = int(cy * view.image.height - viewport_h / 2)
                x2 = int(x1 + viewport_w)
                y2 = int(y1 + viewport_h)

                crop = view.image.crop((x1, y1, x2, y2))
                if fast_resample or (view.image.width > cell_w and view.image.height > cell_h):
                    crop = crop.resize((cell_w, cell_h), Image.Resampling.NEAREST)
                else:
                    crop = crop.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                
                self.canvas_image.paste(crop, (x, y + TEXT_BAR_HEIGHT))
                
            # File name or empty
            fname = os.path.basename(view.path) if view else "(empty)"
            text = f"{LABELS[i]} - {fname}"
            
            if view.comparison_text is not None:
                text += " - " + view.comparison_text

            # Label bar
            draw.rectangle([x, y, x + cell_w, y + TEXT_BAR_HEIGHT], fill=(50, 50, 50))
            draw.text((x + 5, y + 5), text, fill=(255, 255, 255), align="center")


        self.tk_img = ImageTk.PhotoImage(self.canvas_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.root.title(f"View {self.scene_index + 1}/{len(self.scenes)} {self.zoom:.2f}X")


if __name__ == "__main__":
    scenes = load_scenes()
    if len(scenes) == 0:
        print("No matching images found. Check DIRS.")
    else:
        root = tk.Tk()
        app = QuadViewer(root, scenes)
        root.mainloop()