import numpy as np
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
import os

class TextureMapper:
    def __init__(self, mesh, room_textures=None):
        """
        Initialize the texture mapper.
        
        Args:
            mesh: Trimesh object to texture
            room_textures: Dictionary of room textures {room_idx: texture_image}
        """
        self.mesh = mesh
        self.room_textures = room_textures or {}
        self.material_library = {}
        self.texture_dir = "output/3d_textures"
        os.makedirs(self.texture_dir, exist_ok=True)
        
    def apply_textures(self, border_map):
        """Apply textures to the mesh based on the room textures"""
        if not self.room_textures:
            print("No textures provided, using default materials")
            return self.apply_default_materials()
            
        # Create material groups for each room texture
        unique_rooms = np.unique(border_map)
        unique_rooms = unique_rooms[unique_rooms > 0]  # Skip walls
        
        # Process each room texture
        for room_idx in unique_rooms:
            if room_idx not in self.room_textures:
                continue
                
            # Get texture image for this room
            texture_img = self.room_textures[room_idx]
            
            # Save texture as image file
            texture_path = os.path.join(self.texture_dir, f"room_{room_idx}_texture.png")
            if isinstance(texture_img, np.ndarray):
                Image.fromarray(texture_img).save(texture_path)
            else:
                texture_img.save(texture_path)
                
            # Create a material for this texture
            material_name = f"room_{room_idx}_material"
            self.material_library[material_name] = {
                'map_Kd': texture_path
            }
            
            # TODO: Apply material to mesh faces belonging to this room
            # This would require segmenting the mesh by room index
        
        return self.mesh
    
    def apply_default_materials(self):
        """Apply default materials to the mesh"""
        # Create material groups for standard parts
        # Walls material
        walls_material = {
            'Kd': [0.9, 0.9, 0.9],  # White walls
            'Ka': [0.1, 0.1, 0.1],
            'Ks': [0.1, 0.1, 0.1],
            'Ns': 10.0
        }
        
        # Floor material
        floor_material = {
            'Kd': [0.6, 0.4, 0.2],  # Wood-like color
            'Ka': [0.1, 0.1, 0.1],
            'Ks': [0.2, 0.2, 0.2],
            'Ns': 20.0
        }
        
        # Ceiling material
        ceiling_material = {
            'Kd': [0.95, 0.95, 0.95],  # Slightly off-white
            'Ka': [0.1, 0.1, 0.1],
            'Ks': [0.0, 0.0, 0.0],
            'Ns': 0.0
        }
        
        # Furniture materials
        furniture_material = {
            'Kd': [0.4, 0.2, 0.0],  # Dark wood
            'Ka': [0.1, 0.1, 0.1],
            'Ks': [0.3, 0.3, 0.3],
            'Ns': 30.0
        }
        
        # Add materials to library
        self.material_library['walls'] = walls_material
        self.material_library['floor'] = floor_material
        self.material_library['ceiling'] = ceiling_material
        self.material_library['furniture'] = furniture_material
        
        # Simple geometric-based material assignment
        # Assuming Z coordinate can determine material:
        # - Z = 0 -> floor
        # - Z = wall_height -> ceiling
        # - 0 < Z < wall_height and on boundary -> walls
        # - Otherwise -> furniture
        
        wall_height = max(self.mesh.vertices[:, 2])
        
        # Group faces by material
        floor_faces = []
        ceiling_faces = []
        wall_faces = []
        furniture_faces = []
        
        for face_idx, face in enumerate(self.mesh.faces):
            vertices = self.mesh.vertices[face]
            z_values = vertices[:, 2]
            
            # Check z coordinates to determine material
            if np.allclose(z_values, 0):
                floor_faces.append(face_idx)
            elif np.allclose(z_values, wall_height):
                ceiling_faces.append(face_idx)
            elif np.any(np.isclose(z_values, 0)) and np.any(np.isclose(z_values, wall_height)):
                wall_faces.append(face_idx)
            else:
                furniture_faces.append(face_idx)
        
        # Apply materials to mesh groups
        if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'face_materials'):
            self.mesh.visual.face_materials = np.zeros(len(self.mesh.faces), dtype=np.int)
            
            # Assign material indices
            for idx in floor_faces:
                self.mesh.visual.face_materials[idx] = 1  # floor
                
            for idx in ceiling_faces:
                self.mesh.visual.face_materials[idx] = 2  # ceiling
                
            for idx in wall_faces:
                self.mesh.visual.face_materials[idx] = 0  # walls
                
            for idx in furniture_faces:
                self.mesh.visual.face_materials[idx] = 3  # furniture
                
        return self.mesh
    
    def create_texture_atlas(self, texture_size=(1024, 1024)):
        """Create a texture atlas from all room textures"""
        if not self.room_textures:
            return None
            
        # Calculate atlas grid size based on number of textures
        n_textures = len(self.room_textures)
        grid_size = int(np.ceil(np.sqrt(n_textures)))
        
        # Create empty atlas
        atlas = Image.new('RGB', texture_size, (255, 255, 255))
        
        # Calculate individual texture size
        tex_width = texture_size[0] // grid_size
        tex_height = texture_size[1] // grid_size
        
        # Place each texture in the atlas
        for i, (room_idx, texture) in enumerate(self.room_textures.items()):
            row = i // grid_size
            col = i % grid_size
            
            x = col * tex_width
            y = row * tex_height
            
            # Convert texture to PIL if it's numpy
            if isinstance(texture, np.ndarray):
                texture_pil = Image.fromarray(texture.astype(np.uint8))
            else:
                texture_pil = texture
                
            # Resize texture to fit in atlas
            texture_pil = texture_pil.resize((tex_width, tex_height))
            
            # Paste into atlas
            atlas.paste(texture_pil, (x, y))
        
        # Save atlas
        atlas_path = os.path.join(self.texture_dir, "texture_atlas.png")
        atlas.save(atlas_path)
        
        return atlas_path
        
    def export_textured_mesh(self, filename="output/3d_textured_model.obj"):
        """Export the textured mesh to OBJ format with MTL materials"""
        mtl_filename = filename.replace(".obj", ".mtl")
        mtl_basename = os.path.basename(mtl_filename)
        
        # Write MTL file
        with open(mtl_filename, 'w') as f:
            f.write(f"# Material library for {filename}\n\n")
            
            for material_name, material in self.material_library.items():
                f.write(f"newmtl {material_name}\n")
                
                # Write material properties
                if 'Ka' in material:
                    f.write(f"Ka {material['Ka'][0]} {material['Ka'][1]} {material['Ka'][2]}\n")
                if 'Kd' in material:
                    f.write(f"Kd {material['Kd'][0]} {material['Kd'][1]} {material['Kd'][2]}\n")
                if 'Ks' in material:
                    f.write(f"Ks {material['Ks'][0]} {material['Ks'][1]} {material['Ks'][2]}\n")
                if 'Ns' in material:
                    f.write(f"Ns {material['Ns']}\n")
                if 'map_Kd' in material:
                    texture_rel_path = os.path.relpath(material['map_Kd'], os.path.dirname(filename))
                    f.write(f"map_Kd {texture_rel_path}\n")
                
                f.write("\n")
        
        # Export mesh with material reference
        self.mesh.export(filename, include_materials=True, mtl_name=mtl_basename)
        
        return filename 