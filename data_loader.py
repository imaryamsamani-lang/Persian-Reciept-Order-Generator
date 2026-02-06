import io
import ast
from torchvision import transforms
from scipy.ndimage import map_coordinates
import re
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
import os
import random
import cv2

def resize_gray_keep_aspect(gray,
                            min_short_side=384,
                            max_short_side=1024,
                            max_upscale=1.8):
    h, w = gray.shape[:2]
    short = min(h, w)

    if short < min_short_side:
        scale = min(max_upscale, min_short_side / short)
    elif short > max_short_side:
        scale = max_short_side / short
    else:
        scale = 1.0

    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, (new_w, new_h), interpolation=interp)

def pad_to_square(img: np.ndarray, pad_value: int = 255) -> np.ndarray:
    """Pad grayscale or RGB to square without distortion."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    if img.ndim == 2:  # grayscale
        square = np.full((size, size), pad_value, dtype=img.dtype)
        square[:h, :w] = img
    else:  # rgb
        square = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
        square[:h, :w, :] = img
    return square

class DotsOcrJsonl(Dataset):
    def __init__(self, real_data, stopphrases, prompt, numbers, persian_words, category_units, processor, phase):

        self.real_data = real_data

        self.processor = processor
        self.phase = phase

        self.augment = transforms.Compose([transforms.ColorJitter(brightness=0.2,
                                                                  contrast=0.2,
                                                                  saturation=0.2,
                                                                  hue=0.1)])
        self.prompt = prompt
        self.numbers = numbers
        self.persian_words = persian_words
        self.category_units = category_units

        self.bg_paths = os.listdir("backgrounds")

        # Filter list
        self.bg_paths = [x for x in self.bg_paths if any(x.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])]

        # regex for quantity + unit
        self.pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(کیلوگرم|گرم|لیتر|عدد(?:ی)?)')

        self.root_fonts = "fonts/persian_fonts/"
        self.eng_fonts = "fonts/english_fonts/"

        self.shape_x = 1280
        self.shape_y = 960

        self.base_x, self.base_y = 510, 100
        self.variation_x = 0
        self.variation_y = 0

        self.stopphrases = stopphrases

    def __len__(self):
        return len(self.real_data)

    def move_quantity_first(self, s):
        o = random.choice([True, False])
        if o:
          match = self.pattern.search(s)
          if match:
            qty = match.group(0)  # full "200 گرم"
            s = self.pattern.sub('', s, count=1).strip()  # remove first occurrence
            return f"{qty} {s}"
          return s
        else:
          return s

    def _apply_conformal_transformation(self, image_pil, epsilon1=2e-4, epsilon2=0):
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Z_transformed = Z + epsilon1 * Z**2 + epsilon2* Z**3

        # Extract transformed coordinates and shift back
        X_new = np.real(Z_transformed) + w/2
        Y_new = np.imag(Z_transformed) + h/2

        # Flatten coordinate arrays for map_coordinates
        coords = np.vstack([Y_new.flatten(),  # row coordinates
                            X_new.flatten()   # col coordinates
                            ])

        # Prepare output array
        transformed_np = np.zeros_like(image_np)

        if image_np.ndim == 3:
            # Color image: apply map_coordinates for each channel
            for c in range(image_np.shape[2]):
                channel = image_np[..., c]
                transformed_channel = map_coordinates(channel, coords, order=1, mode='reflect')
                transformed_np[..., c] = transformed_channel.reshape(h, w)
        else:
            # Grayscale image
            transformed_np = map_coordinates(image_np, coords, order=1, mode='reflect').reshape(h, w)

        # Convert back to PIL Image
        transformed_pil = Image.fromarray(transformed_np)

        return transformed_pil

    def preprocess(self, phase, bgr, min_short_side=384, max_short_side=1024, clahe_clip=2.0, clahe_grid=(8, 8), sharpen_strength=0.25):

        gray = cv2.cvtColor(np.array(bgr), cv2.COLOR_RGB2GRAY)

        # Contrast enhancement (CLAHE)
        if clahe_clip and clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
            gray = clahe.apply(gray)

        # Mild sharpen
        if sharpen_strength and sharpen_strength > 0:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
            gray = cv2.addWeighted(gray, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)

        # Resize with aspect ratio preserved
        gray = resize_gray_keep_aspect(gray,
                                       min_short_side=min_short_side,
                                       max_short_side=max_short_side)

        # Pad to square to avoid distortion in model
        #gray = pad_to_square(gray, pad_value=255)

        # Convert to 3-channel RGB PIL
        rgb = np.repeat(gray[..., None], 3, axis=2)

        #normalize between 0 to 1
        out = self.augment(Image.fromarray(rgb, mode="RGB"))

        if phase=="train":
          if random.random() < 0.3:
            out = self._apply_conformal_transformation(out)

        return out

    def load_and_downscale(self, img, longest=1280):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = Image.fromarray(img)

        w, h = img.size
        scale = longest / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        return img

    def random_page_color(self, min = 200, max=255):
        # Pick a random light color in range [200–255]
        r = random.randint(min, max)
        g = random.randint(min, max)
        b = random.randint(min, max)
        return (r, g, b)

    def add_random_color(self, img, intensity=0.3):
        """
        Add a random color tint to an image.

        Args:
        img (PIL.Image): input image (RGB or RGBA)
        intensity (float): blending factor between 0 (no effect) and 1 (full color)

        Returns:
        PIL.Image: tinted image
        """
        img = img.convert("RGBA")

        rand_color = (random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255),
                      int(255 * intensity)  # alpha depends on intensity
                      )

        # Create solid color overlay
        overlay = Image.new("RGBA", img.size, rand_color)

        # Blend images
        return Image.alpha_composite(img, overlay)

    def split_runs(self, text):
        # Split into Persian/Arabic vs English/number runs
        # Arabic block: \u0600-\u06FF
        # Extended Arabic (Persian digits, etc.): \u0750-\u077F, \u08A0-\u08FF
        # Basic Latin: a-z, A-Z, 0-9
        pattern = re.compile(r'([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[A-Za-z0-9]+|[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FFA-Za-z0-9]+)')
        return pattern.findall(text)

    def draw_mixed_text(self, draw, x, y, p_text, fill, font_persian, font_english):
        runs = self.split_runs(p_text)

        current_x = x
        for run in runs:
            if re.match(r'^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+$', run):
                font = font_persian
            elif re.match(r'^[A-Za-z0-9]+$', run):
                font = font_english
            else:
                # spaces, punctuation → choose Persian font by default
                font = font_persian

            # Draw the run
            draw.text((current_x, y), run, font=font, fill=fill, anchor="lm")

            # Advance x by run width
            run_width = draw.textlength(run, font=font)
            current_x += run_width

    def detect_language(self, text: str) -> str:
        # Persian letters
        if re.search(r'[\u0600-\u06FF]', text):
            return "Persian"
        # English letters
        elif re.search(r'[A-Za-z]', text):
            return "English"
        else:
            return "Unknown"

    def random_date_like(self, ):
        year = random.choice([f"{random.randint(1400, 1500):04d}", f"{random.randint(00, 99):02d}"])    # 0000–9999, padded to 4 digits
        month = random.choice([f"{random.randint(1, 12):01d}", f"{random.randint(1, 12):02d}"])     # 01–12
        day = random.choice([f"{random.randint(1, 31):01d}", f"{random.randint(1, 31):02d}"])       # 01–28 to stay safe
        return f"{year}/{month}/{day}"

    def generate_data(self, data):

        data = "words"

        N = random.randint(1, 20)  # random size between 3 and 7
        t = random.choice([0,1])

        p = random.choice(self.numbers+self.persian_words)

        sample = []

        for i in range(N):
           ingredient = random.choice(list(self.category_units.keys()))
           unit = random.choice(self.category_units[ingredient])
           s = random.choice([0,1])

           if s==0:
            sample.append(p+" "+unit+" "+ingredient)
           else:
              sample.append(ingredient+" "+p+" "+unit)
              
        if t==0:
          im = Image.new('RGB', (self.shape_y, self.shape_x), color=self.random_page_color()) #'white')
        else:
          im = Image.open("backgrounds/"+random.choice(self.bg_paths)).convert("RGB")
          im = im.resize((self.shape_y, self.shape_x))
          im = self.add_random_color(im, intensity=0.1)

        draw = ImageDraw.Draw(im)

        number = random.choice(['left', 'right', 'none'])
        prepro = random.choice([True, False])

        if number!='left':
          item_list = [self.move_quantity_first(s) for s in sample]
        else:
          item_list = sample

        final_text = ""

        for c, text in enumerate(item_list):

          if number=='left':
            persian_text = str(c+1) + '.' + text
            final_text += persian_text + "\n"

          elif number=='right':
            persian_text =  text + '.' + str(c+1)
            final_text += persian_text + "\n"

          else:
            persian_text = text
            final_text += persian_text + "\n"

          # randomly vary x and y within the range
          if data in ["words"]:
            x = 510 + random.randint(-self.variation_x, self.variation_x)
            y = 300 + 60*c

          else:
            x = self.base_x + random.randint(-self.variation_x, self.variation_x)
            y = self.base_y + 60*c

          if data not in ["kale_english"]:
            if self.detect_language(persian_text)=="Persian":
              pq = [x for x in os.listdir(self.root_fonts) if any(x.lower().endswith(ext) for ext in [".TTF", ".ttf", ".otf"])]
              fnt = self.root_fonts + random.choice(pq)
              size = 1 + random.randint(-self.variation_x, self.variation_x)
              font = ImageFont.truetype(fnt, 30*size)
              draw.text((x, y), persian_text, fill=self.random_page_color(min=0, max=60), font=font, anchor='mm')
            else:
              pq = [x for x in os.listdir(self.eng_fonts) if any(x.lower().endswith(ext) for ext in [".TTF", ".ttf", ".otf"])]
              fnt = self.eng_fonts + random.choice(pq)
              font = ImageFont.truetype(fnt, size=30)
              draw.text((x, y), persian_text, fill=self.random_page_color(min=0, max=60), font=font, anchor='mm')
          else:
              pq = [x for x in os.listdir(self.eng_fonts) if any(x.lower().endswith(ext) for ext in [".TTF", ".ttf", ".otf"])]
              eng_fnt = self.eng_fonts + random.choice(pq)
              eng_font = ImageFont.truetype(eng_fnt, size=30)
              pq = [x for x in os.listdir(self.root_fonts) if any(x.lower().endswith(ext) for ext in [".TTF", ".ttf", ".otf"])]
              per_fnt = self.root_fonts + random.choice(pq)
              size = 1 + random.randint(-self.variation_x, self.variation_x)
              per_font = ImageFont.truetype(per_fnt, 30*size)
              self.draw_mixed_text(draw, x-400, y,
                                   self.reverse_persian_words(persian_text), #self.reverse_by_words(persian_text),
                                   fill=self.random_page_color(min=0, max=60), font_persian=per_font, font_english=eng_font)

        if data in ["words", "single"]:

            ts = 0 + random.randint(-self.variation_x, self.variation_x) #300
            oo = 100 + random.randint(-self.variation_x, self.variation_x)

            selected_phrases = {cat: random.choice(words) for cat, words in self.stopphrases.items()}

            typee = random.choice([0, 1])
            order = random.choice([0, 1])

            pq = [x for x in os.listdir(self.root_fonts) if any(x.lower().endswith(ext) for ext in [".TTF", ".ttf", ".otf"])]
            fnt = self.root_fonts + random.choice(pq)

            size = 1 + random.randint(-self.variation_x, self.variation_x)
            per_font = ImageFont.truetype(fnt, 30*size)

            datee = self.random_date_like()

            draw.text((x-ts, 100), selected_phrases['greeting'], fill='black', font=per_font, anchor='mm')
            if typee==0:
              if order == 0:
                draw.text((x-ts, 150), selected_phrases['request_instruction'], fill='black', font=per_font, anchor='mm')
                draw.text((x-ts, 200), selected_phrases['time_day'], fill='black', font=per_font, anchor='mm')
                if selected_phrases['date'] !='':
                  draw.text((x-ts, 250), selected_phrases['date'] + ' ' + datee, fill='black', font=per_font, anchor='mm')
              else:
                draw.text((x-ts, 1050-oo), selected_phrases['request_instruction'], fill='black', font=per_font, anchor='mm')
                draw.text((x-ts, 1100-oo), selected_phrases['time_day'], fill='black', font=per_font, anchor='mm')
                if selected_phrases['date'] !='':
                  draw.text((x-ts, 1150-oo), selected_phrases['date'] + ' ' + datee, fill='black', font=per_font, anchor='mm')
            else:
              if order == 0:
                draw.text((x-ts, 150), selected_phrases['question'], fill='black', font=per_font, anchor='mm')
                draw.text((x-ts, 200), selected_phrases['time_day'], fill='black', font=per_font, anchor='mm')
                if selected_phrases['date'] !='':
                  draw.text((x-ts, 250), selected_phrases['date']+ ' ' + datee, fill='black', font=per_font, anchor='mm')
              else:
                draw.text((x-ts, 1050-oo), selected_phrases['question'], fill='black', font=per_font, anchor='mm')
                draw.text((x-ts, 1100-oo), selected_phrases['time_day'], fill='black', font=per_font, anchor='mm')
                if selected_phrases['date'] !='':
                  draw.text((x-ts, 1150-oo), selected_phrases['date']+ ' ' + datee, fill='black', font=per_font, anchor='mm')

            draw.text((x-ts, 1200-oo), selected_phrases['thanks'], fill='black', font=per_font, anchor='mm')

            if typee==0:
               if order==0:
                 if selected_phrases['date'] !='':
                   final_text = selected_phrases['greeting'] + "\n" + selected_phrases['request_instruction'] + "\n" + selected_phrases['time_day'] + "\n" + datee + ' ' + selected_phrases['date'] + "\n" + final_text + "\n" + selected_phrases['thanks']
                 else:
                   final_text = selected_phrases['greeting'] + "\n" + selected_phrases['request_instruction'] + "\n" + selected_phrases['time_day'] + "\n" + final_text + "\n" + selected_phrases['thanks']
               else:
                 if selected_phrases['date'] !='':
                   final_text = selected_phrases['greeting'] + "\n" + final_text + "\n" + selected_phrases['request_instruction'] + "\n" + selected_phrases['time_day'] + "\n" + datee + ' ' + selected_phrases['date'] + "\n" + selected_phrases['thanks']
                 else:
                   final_text = selected_phrases['greeting'] + "\n" + final_text + "\n" + selected_phrases['request_instruction'] + "\n" + selected_phrases['time_day'] + "\n" + selected_phrases['thanks']
            else:
               if order==0:
                 if selected_phrases['date'] !='':
                   final_text = selected_phrases['greeting'] + "\n" + selected_phrases['question'] + "\n" + selected_phrases['time_day'] + "\n" + datee + ' ' + selected_phrases['date'] + "\n" + final_text + "\n" + selected_phrases['thanks']
                 else:
                   final_text = selected_phrases['greeting'] + "\n" + selected_phrases['question'] + "\n" + selected_phrases['time_day'] + "\n" + final_text + "\n" + selected_phrases['thanks']
               else:
                 if selected_phrases['date'] !='':
                   final_text = selected_phrases['greeting'] + "\n" + final_text + "\n" + selected_phrases['question'] + "\n" + selected_phrases['time_day'] + "\n" + datee + ' ' + selected_phrases['date'] + "\n" + selected_phrases['thanks']
                 else:
                   final_text = selected_phrases['greeting'] + "\n" + final_text + "\n" + selected_phrases['question'] + "\n" + selected_phrases['time_day'] + "\n" + selected_phrases['thanks']

        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return im, final_text

    def reverse_persian_words(self, text):
        # Ensure numbering like "1.پمینا" → "1. پمینا"
        text = re.sub(r'(\d+\.)(?=[\u0600-\u06FF])', r'\1 ', text)

        tokens = text.split()
        persian_tokens = []
        result_tokens = []

        def flush_persian():
            nonlocal persian_tokens
            if persian_tokens:
                result_tokens.extend(persian_tokens[::-1])
                persian_tokens = []

        for tok in tokens:
            if re.fullmatch(r'\d+\.', tok):  # pure "1." / "2." etc.
                flush_persian()              # stop Persian reversal before it
                result_tokens.append(tok)    # keep numbering in place
            elif re.search(r'[\u0600-\u06FF]', tok):  # Persian word
                persian_tokens.append(tok)
            else:  # English / numbers
                flush_persian()
                result_tokens.append(tok)

        flush_persian()
        return " ".join(result_tokens).strip()

    def __getitem__(self, idx):

        if self.phase != "test":
          data = "synthetic"
        else:
          data = "real"

        if data == "synthetic":
          im, answer = self.generate_data(data)

        else:

          if self.phase == "test":
            sample_dataframe = self.real_data[self.real_data["root"].isin(self.kale_roots)]
            row = sample_dataframe.sample(n=1)

          else:
            row = self.real_data.sample(n=1)

          if row["root"].values[0] != "bytes":
            path = row["root"].values[0]+row["image_path"].values[0]
            im = cv2.imread(path)

          else:
            path = row["image_path"].values[0]
            im = np.array(Image.open(io.BytesIO(ast.literal_eval(path)['bytes'])))

          answer = row["text"].values[0]

        rot = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 90, 90, -90, -90, 180])
        if rot==90:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot==-90:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot==180:
            im = cv2.rotate(im, cv2.ROTATE_180)

        raw_image = self.load_and_downscale(im, longest=560)

        input_img = self.preprocess(self.phase,
                                    raw_image,
                                    min_short_side=384,
                                    max_short_side=1024,
                                    clahe_clip=2.0,
                                    clahe_grid=(8,8),
                                    sharpen_strength=0.25)

        messages_full = [{"role": "user",
                          "content": [{"type": "image", "image": input_img},
                                      {"type": "text", "text": self.prompt}]},
                        {"role": "assistant", "content": answer}]

        text_full   = ""#self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        prompt_only = "" #self.processor.apply_chat_template([messages_full[0]], tokenize=False, add_generation_prompt=True)

        return {"text_full": text_full,
                "prompt_only": prompt_only,
                "answer": answer,
                "image": input_img}