import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import os
import re
import openai
import json

class TextureGenerator:
    def __init__(self, description, border_map, room_name_dict, object_positions=None):
        """
        Initialize the texture generator.
        
        Args:
            description: Text description of the house
            border_map: The border map with room indices
            room_name_dict: Dictionary mapping room names to room types
            object_positions: Dictionary of object positions for each room
        """
        self.description = description
        self.border_map = border_map
        self.room_name_dict = room_name_dict
        self.object_positions = object_positions if object_positions else {}
        self.textures = {}
        
        # Color palettes for different room types
        self.color_palettes = {
            'living_room': [(240, 248, 255), (245, 245, 245), (230, 230, 250)],  # Light blue, white, lavender
            'bedroom': [(255, 240, 245), (255, 228, 225), (250, 235, 215)],  # Pink, misty rose, almond
            'bathroom': [(220, 220, 220), (230, 230, 250), (240, 255, 255)],  # Silver, lavender, azure
            'kitchen': [(255, 250, 240), (255, 255, 224), (245, 245, 220)],  # Floral white, light yellow, beige
            'dining_room': [(250, 240, 230), (255, 228, 196), (255, 222, 173)],  # Linen, bisque, navajo white
            'study_room': [(240, 230, 140), (255, 250, 205), (238, 232, 170)],  # Khaki, lemon chiffon, pale goldenrod
            'storage': [(211, 211, 211), (220, 220, 220), (192, 192, 192)],  # Light gray, silver, silver
            'entrance': [(240, 248, 255), (245, 245, 245), (230, 230, 250)],  # Light blue, white, lavender
            'balcony': [(240, 255, 240), (240, 255, 255), (245, 255, 250)],  # Honeydew, azure, mint cream
            'unknown': [(245, 245, 245), (220, 220, 220), (211, 211, 211)]  # White, silver, light gray
        }
        
        # Floor textures for different room types
        self.floor_textures = {
            'living_room': 'wood',
            'bedroom': 'carpet',
            'bathroom': 'tile',
            'kitchen': 'tile',
            'dining_room': 'wood',
            'study_room': 'wood',
            'storage': 'concrete',
            'entrance': 'wood',
            'balcony': 'concrete',
            'unknown': 'wood'
        }
    
    def query_llm_for_appearances(self):
        """Query LLM to get appearance descriptions for each room"""
        prompt_template = """
        Task: You are an interior designer crafting the appearance for a house described as: {description}
        
        For each room in the house, provide a brief description of the visual appearance including wall colors, 
        flooring type, and overall style. Be specific and match the style to the overall house description.
        
        Rooms to describe: {rooms}
        
        Output the result as a valid JSON object where keys are room names and values are appearance descriptions:
        {{
            "room_name1": "appearance description",
            "room_name2": "appearance description",
            ...
        }}
        """
        
        rooms = list(self.room_name_dict.keys())
        prompt = prompt_template.format(description=self.description, rooms=rooms)
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content
        # Find JSON block using a simpler regex pattern
        pattern = r'\{[\s\S]*\}'  # Match anything between { and }
        json_match = re.search(pattern, response_text)
        
        if json_match:
            try:
                appearance_dict = json.loads(json_match.group())
                return appearance_dict
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response")
                return {room: f"Standard {self.room_name_dict[room]} style" for room in rooms}
        else:
            print("No JSON found in LLM response")
            return {room: f"Standard {self.room_name_dict[room]} style" for room in rooms}
    
    def _generate_wood_texture(self, width, height, color_base=(200, 170, 120)):
        """Generate a simple wood texture"""
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base color with some variation
        for y in range(height):
            row_variation = random.randint(-10, 10)
            for x in range(width):
                pixel_variation = random.randint(-5, 5)
                color = (
                    max(0, min(255, color_base[0] + row_variation + pixel_variation)),
                    max(0, min(255, color_base[1] + row_variation + pixel_variation)),
                    max(0, min(255, color_base[2] + row_variation + pixel_variation))
                )
                texture[y, x] = color
                
        # Add wood grain
        num_grains = random.randint(5, 15)
        for _ in range(num_grains):
            grain_y = random.randint(0, height-1)
            grain_width = random.randint(1, 3)
            grain_color = (
                max(0, min(255, color_base[0] - 30)),
                max(0, min(255, color_base[1] - 30)),
                max(0, min(255, color_base[2] - 30))
            )
            
            grain_y_end = min(grain_y + grain_width, height)
            texture[grain_y:grain_y_end, :] = grain_color
            
        return texture
    
    def _generate_carpet_texture(self, width, height, color_base=(230, 230, 230)):
        """Generate a simple carpet texture"""
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base color
        for y in range(height):
            for x in range(width):
                pixel_variation = random.randint(-15, 15)
                color = (
                    max(0, min(255, color_base[0] + pixel_variation)),
                    max(0, min(255, color_base[1] + pixel_variation)),
                    max(0, min(255, color_base[2] + pixel_variation))
                )
                texture[y, x] = color
                
        # Add carpet grain noise
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return texture
    
    def _generate_tile_texture(self, width, height, color_base=(200, 200, 200), tile_size=32):
        """Generate a simple tile texture"""
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Alternate tile colors slightly
                alt = (x // tile_size + y // tile_size) % 2
                color_variation = 20 if alt else -20
                
                tile_color = (
                    max(0, min(255, color_base[0] + color_variation + random.randint(-5, 5))),
                    max(0, min(255, color_base[1] + color_variation + random.randint(-5, 5))),
                    max(0, min(255, color_base[2] + color_variation + random.randint(-5, 5)))
                )
                
                # Fill tile
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                texture[y:y_end, x:x_end] = tile_color
                
                # Add grout lines
                grout_width = 2
                grout_color = (100, 100, 100)
                
                if y + tile_size < height:
                    texture[y+tile_size-grout_width:y+tile_size, x:x_end] = grout_color
                    
                if x + tile_size < width:
                    texture[y:y_end, x+tile_size-grout_width:x+tile_size] = grout_color
        
        return texture
    
    def _generate_concrete_texture(self, width, height, color_base=(180, 180, 180)):
        """Generate a simple concrete texture"""
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base color with noise
        for y in range(height):
            for x in range(width):
                pixel_variation = random.randint(-20, 20)
                color = (
                    max(0, min(255, color_base[0] + pixel_variation)),
                    max(0, min(255, color_base[1] + pixel_variation)),
                    max(0, min(255, color_base[2] + pixel_variation))
                )
                texture[y, x] = color
        
        # Add some cracks
        num_cracks = random.randint(3, 8)
        for _ in range(num_cracks):
            start_x = random.randint(0, width-1)
            start_y = random.randint(0, height-1)
            length = random.randint(20, 100)
            
            x, y = start_x, start_y
            for i in range(length):
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                x += dx
                y += dy
                
                if 0 <= x < width and 0 <= y < height:
                    crack_color = (
                        max(0, color_base[0] - 40),
                        max(0, color_base[1] - 40),
                        max(0, color_base[2] - 40)
                    )
                    texture[y, x] = crack_color
        
        return texture
    
    def generate_floor_texture(self, room_type, width, height, appearance_desc=None):
        """Generate floor texture based on room type and appearance description"""
        # Parse appearance description to determine color
        color_base = None
        if appearance_desc:
            # Extract color information from description
            color_words = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'brown', 'gray', 'grey', 'beige', 'tan']
            for color in color_words:
                if color in appearance_desc.lower():
                    if color == 'white':
                        color_base = (235, 235, 235)
                    elif color == 'black':
                        color_base = (50, 50, 50)
                    elif color == 'red':
                        color_base = (200, 100, 100)
                    elif color == 'green':
                        color_base = (100, 180, 100)
                    elif color == 'blue':
                        color_base = (100, 100, 200)
                    elif color == 'yellow':
                        color_base = (230, 220, 100)
                    elif color == 'brown':
                        color_base = (150, 100, 50)
                    elif color in ['gray', 'grey']:
                        color_base = (150, 150, 150)
                    elif color in ['beige', 'tan']:
                        color_base = (210, 180, 140)
                    break
        
        # Determine floor type from appearance or default
        floor_type = self.floor_textures.get(room_type, 'wood')
        if appearance_desc:
            if 'wood' in appearance_desc.lower() or 'wooden' in appearance_desc.lower() or 'hardwood' in appearance_desc.lower():
                floor_type = 'wood'
            elif 'carpet' in appearance_desc.lower() or 'rug' in appearance_desc.lower():
                floor_type = 'carpet'
            elif 'tile' in appearance_desc.lower():
                floor_type = 'tile'
            elif 'concrete' in appearance_desc.lower() or 'stone' in appearance_desc.lower() or 'marble' in appearance_desc.lower():
                floor_type = 'concrete'
        
        # Use default color if none was extracted
        if not color_base:
            if floor_type == 'wood':
                color_base = (200, 170, 120)  # Default wood color
            elif floor_type == 'carpet':
                color_base = (230, 230, 230)  # Default carpet color
            elif floor_type == 'tile':
                color_base = (200, 200, 200)  # Default tile color
            elif floor_type == 'concrete':
                color_base = (180, 180, 180)  # Default concrete color
        
        # Generate texture based on floor type
        if floor_type == 'wood':
            return self._generate_wood_texture(width, height, color_base)
        elif floor_type == 'carpet':
            return self._generate_carpet_texture(width, height, color_base)
        elif floor_type == 'tile':
            return self._generate_tile_texture(width, height, color_base)
        elif floor_type == 'concrete':
            return self._generate_concrete_texture(width, height, color_base)
        else:
            # Default to wood if unknown type
            return self._generate_wood_texture(width, height, color_base)
    
    def generate_wall_texture(self, room_type, width, height, appearance_desc=None):
        """Generate wall texture based on room type and appearance description"""
        # Parse appearance description to determine color
        color_base = None
        if appearance_desc:
            # Extract color information from description for walls
            walls_desc = ""
            if "wall" in appearance_desc.lower():
                wall_idx = appearance_desc.lower().find("wall")
                walls_desc = appearance_desc[wall_idx:wall_idx+50]  # Get subset of text about walls
            else:
                walls_desc = appearance_desc  # Use entire description
                
            color_words = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'brown', 'gray', 'grey', 'beige', 'tan']
            for color in color_words:
                if color in walls_desc.lower():
                    if color == 'white':
                        color_base = (245, 245, 245)
                    elif color == 'black':
                        color_base = (50, 50, 50)
                    elif color == 'red':
                        color_base = (220, 120, 120)
                    elif color == 'green':
                        color_base = (120, 200, 120)
                    elif color == 'blue':
                        color_base = (120, 120, 220)
                    elif color == 'yellow':
                        color_base = (240, 230, 140)
                    elif color == 'brown':
                        color_base = (165, 120, 70)
                    elif color in ['gray', 'grey']:
                        color_base = (200, 200, 200)
                    elif color in ['beige', 'tan']:
                        color_base = (225, 200, 170)
                    break
        
        # Use default color if none was extracted
        if not color_base:
            # Get a color from the palette for this room type
            palette = self.color_palettes.get(room_type, self.color_palettes['unknown'])
            color_base = palette[random.randint(0, len(palette)-1)]
        
        # Create a simple wall texture with some noise
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                pixel_variation = random.randint(-10, 10)
                color = (
                    max(0, min(255, color_base[0] + pixel_variation)),
                    max(0, min(255, color_base[1] + pixel_variation)),
                    max(0, min(255, color_base[2] + pixel_variation))
                )
                texture[y, x] = color
        
        return texture
    
    def render_textured_room(self, room_idx, appearance_desc=None, output_dir="output"):
        """Render a textured version of a room"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mask for this room
        room_mask = (self.border_map == room_idx).astype(np.uint8)
        if not np.any(room_mask):
            print(f"Room index {room_idx} not found in border map")
            return None
            
        # Get room type
        room_name = None
        for name, idx in zip(self.room_name_dict.keys(), range(len(self.room_name_dict))):
            if idx == room_idx:
                room_name = name
                break
                
        if not room_name:
            print(f"Room name not found for index {room_idx}")
            return None
            
        room_type = self.room_name_dict[room_name]
        
        # Get dimensions of the room
        y_indices, x_indices = np.where(room_mask > 0)
        min_y, max_y = min(y_indices), max(y_indices)
        min_x, max_x = min(x_indices), max(x_indices)
        room_height = max_y - min_y + 1
        room_width = max_x - min_x + 1
        
        # Generate floor and wall textures
        floor_texture = self.generate_floor_texture(room_type, room_width, room_height, appearance_desc)
        wall_texture = self.generate_wall_texture(room_type, room_width, room_height, appearance_desc)
        
        # Create a combined texture image for the room
        texture_img = np.zeros((self.border_map.shape[0], self.border_map.shape[1], 3), dtype=np.uint8)
        
        # Fill with the floor texture
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if room_mask[y, x] > 0:
                    texture_img[y, x] = floor_texture[y - min_y, x - min_x]
        
        # Add wall borders (detect edges in the room mask)
        wall_thickness = 3
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if room_mask[y, x] > 0:
                    # Check if this is a border pixel (has a non-room neighbor)
                    is_border = False
                    for dy in range(-wall_thickness, wall_thickness + 1):
                        for dx in range(-wall_thickness, wall_thickness + 1):
                            if (0 <= y + dy < room_mask.shape[0] and 
                                0 <= x + dx < room_mask.shape[1] and
                                room_mask[y + dy, x + dx] == 0):
                                is_border = True
                                break
                        if is_border:
                            break
                            
                    if is_border:
                        texture_img[y, x] = wall_texture[y - min_y, x - min_x]
        
        # Save the texture image
        plt.figure(figsize=(10, 10))
        plt.imshow(texture_img)
        plt.axis('off')
        plt.title(f"{room_name} ({room_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"textured_room_{room_idx}.png"))
        plt.close()
        
        self.textures[room_idx] = texture_img
        return texture_img
        
    def render_all_rooms(self, output_dir="output"):
        """Render textures for all rooms"""
        # Get appearance descriptions from LLM
        appearance_dict = self.query_llm_for_appearances()
        
        textured_rooms = {}
        for room_name, room_type in self.room_name_dict.items():
            # Find room index
            room_idx = list(self.room_name_dict.keys()).index(room_name)
            appearance_desc = appearance_dict.get(room_name)
            
            print(f"Rendering texture for {room_name} (type: {room_type})")
            texture = self.render_textured_room(room_idx, appearance_desc, output_dir)
            if texture is not None:
                textured_rooms[room_name] = texture
        
        # Create combined visualization of all textured rooms
        combined_texture = np.zeros_like(self.border_map, dtype=np.uint8)
        for room_idx, texture in self.textures.items():
            mask = (self.border_map == room_idx)
            combined_texture[mask] = 1
            
        # Convert to RGB
        combined_rgb = np.zeros((self.border_map.shape[0], self.border_map.shape[1], 3), dtype=np.uint8)
        for room_idx, texture in self.textures.items():
            mask = (self.border_map == room_idx)
            combined_rgb[mask] = texture[mask]
        
        # Fill in walls (where border_map is 0) with dark color
        combined_rgb[self.border_map == 0] = [50, 50, 50]
        
        plt.figure(figsize=(12, 12))
        plt.imshow(combined_rgb)
        plt.axis('off')
        plt.title(f"Textured House: {self.description}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "textured_house.png"))
        plt.close()
        
        return combined_rgb 