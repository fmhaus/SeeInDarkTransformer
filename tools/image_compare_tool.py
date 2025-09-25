import os
import json
import math
import tkinter as tk
from tkinter import filedialog, simpledialog
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageTk, ImageDraw

SAVESTATE_FILE = './image_compare_savestate.json'
NUM_WORKERS = 4

IMAGE_GAP = 5
TEXT_BAR_HEIGHT = 20

def is_image_file(filename):
    try:
        return f".{filename.lower().split('.')[-1]}" in Image.registered_extensions()
    except:
        return False

class ImageSource:
    def __init__(self, path, prefix_length):
        self.path = path
        assert os.path.isdir(path)
        label_file = os.path.join(path, 'label.txt')
        if os.path.isfile(label_file):
            with open(label_file, 'r') as fr:
                self.label = fr.read()
        else:
            self.label = os.path.basename(path)
        self.files = {f[:prefix_length]:f for f in os.listdir(path) if is_image_file(f)}
        self.key_list = None
    
    def get_file(self, key):
        return os.path.join(self.path, self.files[key])

class ImageView:
    def __init__(self, path, future):
        self.path = path
        self.future = future
        self.image = None
        self.comparison_text = None
    
    def get_image(self):
        # get image if stored, otherwise wait for future if finished
        if self.image is not None:
            return self.image
        self.image = self.future.result()
        return self.image
    
    def set_gt(self):
        self.comparison_text = 'GT'
    
    def calculate_psnr(self, ground_truth_image):
        im_array = np.frombuffer(self.image.tobytes(), dtype=np.uint8)
        gt_array = np.frombuffer(ground_truth_image.tobytes(), dtype=np.uint8)
        mse = ((im_array - gt_array)**2).mean()
        psnr = 20.0 * math.log10(255) - 10.0 * math.log10(mse)
        self.comparison_text = f'PSNR: {psnr:.2f}dB'

class ImageCompareApp:
    def __init__(self, root):
        self.root = root
        self.redraw_after_resize_callback = None
        
        # use thread pool for parallel and async image loading
        self.loader = ThreadPoolExecutor(4)
        
        # images state
        self.image_sources = [None, None, None, None]
        self.image_views = [None, None, None, None]
        self.key_list = []
        self.key_index = None
        self.gt_index = None
        self.initial_dir = './'
        self.prefix_length = 8

        # zoom and dragging state
        self.dragging = False
        self.zoom = 1.0
        self.crop_center = (0.5, 0.5)
        self.start_drag_crop_center = (0, 0)
        self.start_drag_mouse = (0, 0)

        # configure UI
        root.geometry('1200x850')

        menu_bar = tk.Menu(root)
        root.config(menu=menu_bar)
        
        self.change_view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='Views', menu=self.change_view_menu)
        self.change_view_menu.add_command(label='1 - Empty', command=lambda: self.menu_change_view(0))
        self.change_view_menu.add_command(label='2 - Empty', command=lambda: self.menu_change_view(1))
        self.change_view_menu.add_command(label='3 - Empty', command=lambda: self.menu_change_view(2))
        self.change_view_menu.add_command(label='4 - Empty', command=lambda: self.menu_change_view(3))
        
        self.images_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='Images', menu=self.images_menu)
        
        self.settings_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='Settings', menu=self.settings_menu)
        self.settings_menu.add_command(label=f'Prefix length: {self.prefix_length}', command=lambda: self.menu_enter_prefix_length())
        
        self.set_gt_menu = tk.Menu(menu_bar, tearoff=0)
        self.settings_menu.add_cascade(label='Set Ground-Truth', menu=self.set_gt_menu)
        self.set_gt_menu.add_command(label='1 - Empty', command=lambda: self.menu_set_gt(0), state='disabled')
        self.set_gt_menu.add_command(label='2 - Empty', command=lambda: self.menu_set_gt(1), state='disabled')
        self.set_gt_menu.add_command(label='3 - Empty', command=lambda: self.menu_set_gt(2), state='disabled')
        self.set_gt_menu.add_command(label='4 - Empty', command=lambda: self.menu_set_gt(3), state='disabled')

        self.canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
               
        self.canvas_image = Image.new('RGB', (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.tk_img = None

        # input events
        root.bind('<Right>', self.next_image)
        root.bind('<Left>', self.prev_image)
        self.canvas.bind('<Configure>', self.on_resize)
        self.canvas.bind('<MouseWheel>', self.on_zoom)        # Windows
        self.canvas.bind('<Button-4>', self.on_zoom)          # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_zoom)          # Linux scroll down
        self.canvas.bind('<ButtonPress-1>', self.start_drag)
        self.canvas.bind('<B1-Motion>', self.do_drag)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drag)

        # load savestate
        if os.path.isfile(SAVESTATE_FILE):
            with open(SAVESTATE_FILE, 'r') as fr:
                settings = json.loads(fr.read())
                
            if 'sources' in settings:
                if 'prefix_length' in settings and type(settings['prefix_length']) == int:
                    self.prefix_length = settings['prefix_length']
                    self.settings_menu.entryconfigure(0, label=f'Prefix length: {self.prefix_length}')
                
                for i, path in enumerate(settings['sources']):
                    if os.path.isdir(path):
                        self.image_sources[i] = ImageSource(path, self.prefix_length)
                
                self.update_key_list([i for i in range(4) if self.image_sources[i]])
                if 'key_index' in settings and type(settings['key_index']) == int and len(self.key_list) > 0:
                    self.key_index = min(int(settings['key_index']), len(self.key_list))
                if 'gt_index' in settings and type(settings['gt_index']) == int:
                    self.gt_index = settings['gt_index']

                self.load_images()

        self.redraw()

    def on_resize(self, _):
        # recreate canvas 100ms after resize is ended to prevent repeated canvas creations
        if self.redraw_after_resize_callback is not None:
            self.root.after_cancel(self.redraw_after_resize_callback)
        self.redraw_after_resize_callback = self.root.after(100, self.after_resize)
        
    def after_resize(self):
        # do actual resize here
        self.redraw_after_resize_callback = None
        self.canvas_image = Image.new('RGB', (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.redraw()
    
    def save_settings(self):
        settings = {
            'sources': [source.path if source else '' for source in self.image_sources],
            'key_index': self.key_index,
            'gt_index': self.gt_index,
            'prefix_length': self.prefix_length
        }
        with open(SAVESTATE_FILE, 'w') as fw:
            fw.write(json.dumps(settings))
    
    def menu_change_view(self, index):
        path = filedialog.askdirectory(title='Select view folder', initialdir=self.initial_dir)
        if not path:
            return
        
        self.image_sources[index] = ImageSource(path, self.prefix_length)
        self.initial_dir = os.path.dirname(path)
        if self.gt_index == index:
            self.gt_index = None
        
        self.update_key_list([index])
        self.load_images()
        self.redraw()
        self.save_settings()
    
    def menu_go_to_key(self, index):
        self.key_index = index
        self.load_images()
        self.redraw()
        self.save_settings()
    
    def menu_enter_prefix_length(self):
        result = simpledialog.askinteger(
            "Prefix length",
            "Enter new prefix length",
            minvalue=1,
            maxvalue=100
            )

        if result is None:
            return
        
        self.prefix_length = result
        self.settings_menu.entryconfigure(0, label=f'Prefix length: {self.prefix_length}')
        
        # recreate sources with new prefix
        self.image_sources = [ImageSource(source.path, self.prefix_length) if source else None for source in self.image_sources]
        self.update_key_list([i for i in range(4) if self.image_sources[i]])
        self.load_images()
        self.redraw()
        self.save_settings()
    
    def menu_set_gt(self, index):
        self.gt_index = index
        
        for i, view in enumerate(self.image_views):
            if view:
                if i == self.gt_index:
                    view.set_gt()
                elif self.image_views[self.gt_index] and self.image_views[self.gt_index].image:
                    view.calculate_psnr(self.image_views[self.gt_index].image)
        
        self.redraw(redraw_images=[])
        self.save_settings()
    
    def update_key_list(self, update_indices):
        # find all common keys from available sources
        available_sources = [source for source in self.image_sources if source is not None]
        common_keys = set.intersection(*(set(source.files.keys()) for source in available_sources))
        self.key_list = list(common_keys)
        self.key_list.sort()
        
        # make sure key index is within bounds
        if len(self.key_list) == 0:
            self.key_index = None
        else:
            if self.key_index is None:
                self.key_index = 0
            else:
                self.key_index = min(self.key_index, len(self.key_list) - 1)
        
        # update menus
        for index in update_indices:
            label = f'{index+1}: {self.image_sources[index].label}'
            self.change_view_menu.entryconfigure(index, label=label)
            self.set_gt_menu.entryconfigure(index, label=label)
            self.set_gt_menu.entryconfigure(index, state='normal')

        self.images_menu.delete(0, tk.END)
        for i, key in enumerate(self.key_list):
            self.images_menu.add_command(label=key, command=lambda idx=i: self.menu_go_to_key(idx))

    def load_images(self):
        # cancel remaining futures
        for view in self.image_views:
            if view and not view.image:
                view.future.cancel()
        
        if self.key_index is None:
            self.image_views = [None, None, None, None]
            return
        
        # submit loads
        key = self.key_list[self.key_index]
        for i, source in enumerate(self.image_sources):
            if source:
                path = source.get_file(key)
                future = self.loader.submit(self.load_image_worker_thread, path, self.key_index, i)
                self.image_views[i] = ImageView(path, future)
        
        # resets transformations
        self.zoom = 1.0
        self.crop_center = (0.5, 0.5)
        self.dragging = False
    
    def load_image_worker_thread(self, path, key_index, image_index):
        # called by thread executor
        try:
            image = Image.open(path)
            image.load()
            self.root.after(75, self.on_image_load_main_thread, key_index, image_index)
            return image
        except:
            return None
    
    def on_image_load_main_thread(self, key_index, image_index):
        if self.key_index != key_index:
            return
        # pull image from future
        try:
            self.image_views[image_index].get_image()
        except:
            return
        
        if image_index == self.gt_index:
            # calculate comparisons for all images
            for i, view in enumerate(self.image_views):
                if view:
                    if i == image_index:
                        view.set_gt()
                    elif view.image:
                        view.calculate_psnr(self.image_views[self.gt_index].image)
        else:
            # calculate comparison for new image
            if self.gt_index is not None:
                gt_view = self.image_views[self.gt_index]
                if gt_view and gt_view.image:
                    self.image_views[image_index].calculate_psnr(gt_view.image)
        
       
        self.redraw([image_index])
        
    def next_image(self, _event=None):
        self.key_index = (self.key_index + 1) % len(self.key_list)
        self.load_images()
        self.redraw()
        self.save_settings()
            
    def prev_image(self, _event=None):
        self.key_index = (self.key_index + len(self.key_list) - 1) % len(self.key_list)
        self.load_images()
        self.redraw()
        self.save_settings()

    def on_zoom(self, event):
        delta = 0
        if event.num == 4: 
            delta = 1
        elif event.num == 5:
            delta = -1
        elif event.delta:
            delta = 1 if event.delta > 0 else -1

        factor = 1.1 if delta > 0 else 0.9
        self.zoom = max(1.0, min(self.zoom * factor, 100.0))  # clamp
        
        half_crop_size = 0.5 / self.zoom
        cx, cy = self.crop_center
        # clamp center to image bounds
        cx = min(max(cx, half_crop_size), 1.0 - half_crop_size)
        cy = min(max(cy, half_crop_size), 1.0 - half_crop_size)
        self.crop_center = (cx, cy)
        
        self.redraw()
    
    def start_drag(self, event):
        self.dragging = True
        self.start_drag_mouse = (event.x, event.y)
        self.start_drag_crop_center = self.crop_center
        
    def stop_drag(self, _):
        self.dragging = False
        self.redraw()

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
        self.redraw()

    def current_canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        return w, h
    
    def current_cell_size(self):
        canvas_w, canvas_h = self.current_canvas_size()
        w = (canvas_w - IMAGE_GAP) // 2
        h = (canvas_h - IMAGE_GAP - 2 * TEXT_BAR_HEIGHT) // 2
        return w, h
    
    def current_cell_dim(self, index):
        w, h = self.current_cell_size()
        row, col = divmod(index, 2)
        x = col * (w + IMAGE_GAP)
        y = row * (h + IMAGE_GAP + TEXT_BAR_HEIGHT)
        return x, y, w, h
        

    def redraw(self, redraw_images=[0, 1, 2, 3]):
        w, h = self.current_canvas_size()
        if w <= 2 or h <= 2:
            return
        
        draw = ImageDraw.Draw(self.canvas_image)
        
        for i in redraw_images:
            view = self.image_views[i]
            x, y, w, h = self.current_cell_dim(i)

            if view and view.image:
                image = view.image
                    
                # Compute crop
                viewport_w = image.width / self.zoom
                viewport_h = image.height / self.zoom
                cx, cy = self.crop_center
                x1 = int(cx * image.width - viewport_w / 2)
                y1 = int(cy * image.height - viewport_h / 2)
                x2 = int(x1 + viewport_w)
                y2 = int(y1 + viewport_h)

                crop = image.crop((x1, y1, x2, y2))
                crop = crop.resize((w, h), Image.Resampling.NEAREST)
                
                self.canvas_image.paste(crop, (x, y + TEXT_BAR_HEIGHT))

            else:
                draw.rectangle([x, y + TEXT_BAR_HEIGHT, x + w, y + TEXT_BAR_HEIGHT + h], fill="black")

        for i, view in enumerate(self.image_views):
            if view:
                text = f'{self.image_sources[i].label} - {os.path.basename(view.path)}'
            
                if view.comparison_text:    
                    text += ' - ' + view.comparison_text
            else:
                text = '(empty)'
            
            x, y, w, h = self.current_cell_dim(i)
            draw.rectangle([x, y, x + w, y + TEXT_BAR_HEIGHT], fill=(50, 50, 50))
            draw.text((x + 5, y + 5), text, fill=(255, 255, 255), align='center')

        self.tk_img = ImageTk.PhotoImage(self.canvas_image)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

        if len(self.key_list) != 0:
            self.root.title(f'View {self.key_index + 1}/{len(self.key_list)} {self.zoom:.2f}X')
        else:
            self.root.title(f'Empty')


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageCompareApp(root)
    root.mainloop()