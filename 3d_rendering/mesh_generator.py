import numpy as np
import trimesh
import os
import traceback
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

class MeshGenerator:
    def __init__(self, house_v, house_f, border_map, room_name_dict):
        """
        Initialize the mesh generator.
        
        Args:
            house_v: House vertices
            house_f: House faces
            border_map: Border map of the house
            room_name_dict: Dictionary mapping room names to room types
        """
        self.house_v = house_v
        self.house_f = house_f
        self.border_map = border_map
        self.room_name_dict = room_name_dict
        self.wall_height = 3.0  # Standard wall height in meters
        self.floor_height = 0.1  # Floor thickness in meters
        self.furniture_height = 0.5  # Default furniture height
        self.mesh = None
        
        # Debug info
        print(f"MeshGenerator initialized with {len(self.house_v)} vertices and {len(self.house_f)} faces")
        print(f"Border map shape: {self.border_map.shape}")
        print(f"Room names: {self.room_name_dict}")
        
    def generate_room_mesh(self, room_idx, add_ceiling=True):
        """Generate a 3D mesh for a specific room"""
        room_mask = (self.border_map == room_idx).astype(np.uint8)
        
        # Erode and dilate to get walls
        inner_mask = binary_erosion(room_mask, iterations=1)
        wall_mask = room_mask - inner_mask
        
        # Get wall vertices and faces
        wall_verts = []
        wall_faces = []
        face_count = 0
        
        # Process the mask to create walls
        h, w = room_mask.shape
        for y in range(h):
            for x in range(w):
                if wall_mask[y, x] == 1:
                    # Add vertices for a wall segment (bottom and top)
                    wall_verts.extend([
                        [x, y, 0],                  # Bottom SW
                        [x+1, y, 0],                # Bottom SE
                        [x+1, y+1, 0],              # Bottom NE
                        [x, y+1, 0],                # Bottom NW
                        [x, y, self.wall_height],   # Top SW
                        [x+1, y, self.wall_height], # Top SE
                        [x+1, y+1, self.wall_height], # Top NE
                        [x, y+1, self.wall_height]  # Top NW
                    ])
                    
                    # Add faces for wall segment (6 faces for a cube)
                    idx = face_count * 8
                    # Bottom face
                    wall_faces.append([idx, idx+1, idx+2])
                    wall_faces.append([idx, idx+2, idx+3])
                    # Top face
                    wall_faces.append([idx+4, idx+6, idx+5])
                    wall_faces.append([idx+4, idx+7, idx+6])
                    # Side faces
                    wall_faces.append([idx, idx+4, idx+1])
                    wall_faces.append([idx+1, idx+4, idx+5])
                    wall_faces.append([idx+1, idx+5, idx+2])
                    wall_faces.append([idx+2, idx+5, idx+6])
                    wall_faces.append([idx+2, idx+6, idx+3])
                    wall_faces.append([idx+3, idx+6, idx+7])
                    wall_faces.append([idx+3, idx+7, idx+0])
                    wall_faces.append([idx+0, idx+7, idx+4])
                    
                    face_count += 1
        
        # Create floor
        floor_verts = []
        floor_faces = []
        face_idx = 0
        
        for y in range(h):
            for x in range(w):
                if inner_mask[y, x] == 1:
                    # Add vertices for floor
                    floor_verts.extend([
                        [x, y, 0],
                        [x+1, y, 0],
                        [x+1, y+1, 0],
                        [x, y+1, 0]
                    ])
                    
                    # Add faces for floor
                    idx = face_idx * 4
                    floor_faces.append([idx, idx+1, idx+2])
                    floor_faces.append([idx, idx+2, idx+3])
                    
                    face_idx += 1
        
        # Combine walls and floor
        all_verts = np.array(wall_verts + floor_verts)
        all_faces = np.array(wall_faces + [[f[0] + len(wall_verts), f[1] + len(wall_verts), f[2] + len(wall_verts)] for f in floor_faces])
        
        # Add ceiling if requested
        if add_ceiling:
            ceiling_verts = []
            ceiling_faces = []
            face_idx = 0
            
            for y in range(h):
                for x in range(w):
                    if inner_mask[y, x] == 1:
                        # Add vertices for ceiling (at wall height)
                        ceiling_verts.extend([
                            [x, y, self.wall_height],
                            [x+1, y, self.wall_height],
                            [x+1, y+1, self.wall_height],
                            [x, y+1, self.wall_height]
                        ])
                        
                        # Add faces for ceiling
                        idx = face_idx * 4
                        ceiling_faces.append([idx, idx+2, idx+1])  # Note: reversed winding compared to floor
                        ceiling_faces.append([idx, idx+3, idx+2])
                        
                        face_idx += 1
            
            # Append ceiling to combined mesh
            ceiling_offset = len(all_verts)
            all_verts = np.vstack([all_verts, ceiling_verts])
            all_faces = np.vstack([all_faces, [[f[0] + ceiling_offset, f[1] + ceiling_offset, f[2] + ceiling_offset] for f in ceiling_faces]])
        
        # Create mesh
        room_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        return room_mesh
    
    def generate_house_mesh(self):
        """Generate a basic 3D mesh for the house with walls and floors"""
        try:
            print("Generating house mesh...")
            
            # Convert house vertices to 3D by adding z coordinate
            v_3d = np.zeros((len(self.house_v), 3))
            
            # Check if self.house_v already has 3 dimensions
            if self.house_v.shape[1] == 3:
                v_3d = self.house_v.copy()  # Already 3D, just copy
            else:
                v_3d[:, 0:2] = self.house_v  # Copy x, y coordinates
                v_3d[:, 2] = 0.0  # Set z to 0 for the floor level
            
            # Create the base floor mesh using the existing faces
            floor_mesh = trimesh.Trimesh(vertices=v_3d, faces=self.house_f)
            
            # Get unique room indices
            room_indices = np.unique(self.border_map)
            room_indices = room_indices[room_indices > 0]  # Skip walls and outside
            
            # Generate mesh for each room
            room_meshes = []
            for idx in room_indices:
                room_mesh = self.generate_room_mesh(idx)
                room_meshes.append(room_mesh)
            
            # Combine all meshes
            if room_meshes:
                self.mesh = trimesh.util.concatenate(room_meshes)
            else:
                # Fallback to using the provided house_v and house_f
                print("No room meshes generated, using house_v and house_f directly")
                self.mesh = trimesh.Trimesh(vertices=self.house_v, faces=self.house_f)
            
            return self.mesh
            
        except Exception as e:
            print(f"Error generating house mesh: {e}")
            traceback.print_exc()
            # Create a fallback simple box as the house
            self.mesh = trimesh.creation.box(extents=[10, 10, 3])
            self.mesh.apply_translation([5, 5, 1.5])
            self.mesh.visual.face_colors = [200, 200, 220, 255]
            return self.mesh
    
    def add_furniture_to_mesh(self, all_pos, all_siz, all_ang):
        """Add furniture to the house mesh based on the layout"""
        try:
            if self.mesh is None:
                self.mesh = self.generate_house_mesh()
            
            # Track all furniture meshes
            furniture_meshes = []
            
            # Add a ground plane to ensure visibility
            ground = trimesh.creation.box(extents=[100, 100, 0.1])
            ground.apply_translation([0, 0, -0.1])
            ground.visual.face_colors = [100, 100, 100, 255]  # Dark gray
            furniture_meshes.append(ground)
            
            # Iterate through each room
            for room_idx, (positions, sizes, angles) in enumerate(zip(all_pos, all_siz, all_ang)):
                if not positions:  # Skip empty rooms
                    continue
                
                print(f"Adding furniture for room {room_idx} with {len(positions)} objects")
                
                # Add furniture for this room
                for obj_name, pos in positions.items():
                    try:
                        # Ensure pos is valid
                        if pos is None or len(pos) < 2:
                            print(f"Warning: Invalid position for {obj_name} in room {room_idx}")
                            continue
                            
                        # Get furniture size (or use default)
                        size = sizes.get(obj_name, [1.0, 1.0, self.furniture_height])
                        
                        # Ensure we have 3D size
                        if len(size) < 3:
                            size = list(size) + [self.furniture_height]
                            
                        # Get angle (or use 0)
                        ang = angles.get(obj_name, 0) if angles else 0
                        
                        # Convert string angles to numeric (e.g., 'N', 'S', 'E', 'W')
                        if isinstance(ang, str):
                            ang_map = {'N': 0, 'E': 90, 'S': 180, 'W': 270, 
                                       'NE': 45, 'SE': 135, 'SW': 225, 'NW': 315}
                            ang = ang_map.get(ang, 0)
                        
                        # Apply rotation (around z-axis)
                        rotation = trimesh.transformations.rotation_matrix(
                            angle=np.radians(float(ang)),
                            direction=[0, 0, 1],
                            point=[0, 0, 0]
                        )
                        
                        # Create furniture mesh
                        furniture_box = trimesh.creation.box(size)
                        
                        # Apply rotation to furniture mesh
                        furniture_box.apply_transform(rotation)
                        
                        # Apply translation to furniture mesh
                        furniture_box.apply_translation(pos)
                        
                        # Add to list
                        furniture_meshes.append(furniture_box)
                        
                    except Exception as e:
                        print(f"Error adding furniture {obj_name} in room {room_idx}: {e}")
                        traceback.print_exc()
            
            # Combine with main mesh
            if furniture_meshes:
                self.mesh = trimesh.util.concatenate([self.mesh] + furniture_meshes)
            
            return self.mesh
            
        except Exception as e:
            print(f"Error adding furniture to mesh: {e}")
            traceback.print_exc()
            # Rest of the function remains unchanged
    
    def export_mesh(self, filename="output/3d_model.obj"):
        """Export the mesh to an OBJ file"""
        if self.mesh is None:
            self.generate_house_mesh()
            
        self.mesh.export(filename)
        return filename
    
    def visualize_mesh(self):
        """Visualize the mesh using matplotlib"""
        if self.mesh is None:
            self.generate_house_mesh()
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh faces
        for face in self.mesh.faces:
            vertices = self.mesh.vertices[face]
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            ax.plot_trisurf(x, y, z, color='gray', alpha=0.8)
        
        # Set plot limits and labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D House Mesh')
        
        plt.tight_layout()
        plt.savefig("output/3d_mesh_visualization.png")
        plt.show() 