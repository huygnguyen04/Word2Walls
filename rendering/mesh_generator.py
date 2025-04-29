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
        # Convert lists to numpy arrays if needed
        self.house_v = np.array(house_v) if not isinstance(house_v, np.ndarray) else house_v
        self.house_f = np.array(house_f) if not isinstance(house_f, np.ndarray) else house_f
        self.border_map = border_map
        self.room_name_dict = room_name_dict
        self.wall_height = 3.0  # Standard wall height in meters
        self.floor_height = 0.1  # Floor thickness in meters
        self.furniture_height = 0.8  # Increase default furniture height for better visibility
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
        if not wall_verts and not floor_verts:
            # If no vertices were created, return an empty mesh
            print(f"Warning: Room {room_idx} has no vertices. Creating an empty mesh.")
            return trimesh.Trimesh()
            
        # Handle case where there might be no wall vertices
        if wall_verts and floor_verts:
            all_verts = np.array(wall_verts + floor_verts)
            all_faces = np.array(wall_faces + [[f[0] + len(wall_verts), f[1] + len(wall_verts), f[2] + len(wall_verts)] for f in floor_faces])
        elif wall_verts:
            all_verts = np.array(wall_verts)
            all_faces = np.array(wall_faces)
        elif floor_verts:
            all_verts = np.array(floor_verts)
            all_faces = np.array(floor_faces)
        else:
            # Should be caught by earlier check, but just in case
            return trimesh.Trimesh()
        
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
            
            # Only add ceiling if we have vertices
            if ceiling_verts:
                ceiling_offset = len(all_verts)
                ceiling_verts_array = np.array(ceiling_verts)
                ceiling_faces_array = np.array([[f[0] + ceiling_offset, f[1] + ceiling_offset, f[2] + ceiling_offset] for f in ceiling_faces])
                
                all_verts = np.vstack([all_verts, ceiling_verts_array])
                all_faces = np.vstack([all_faces, ceiling_faces_array])
            else:
                print(f"Warning: Room {room_idx} has no ceiling vertices.")
        
        # Create mesh
        try:
            room_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        except Exception as e:
            print(f"Error creating mesh for room {room_idx}: {e}")
            room_mesh = trimesh.Trimesh()
            
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
                
            # Validate face indices to ensure they don't exceed vertex count
            valid_faces = []
            max_vertex_idx = len(v_3d) - 1
            
            for face in self.house_f:
                if all(idx <= max_vertex_idx for idx in face):
                    valid_faces.append(face)
                else:
                    print(f"Warning: Skipping face with invalid indices: {face}")
            
            if len(valid_faces) == 0:
                print("No valid faces found. Creating a fallback mesh.")
                # Create a simple fallback mesh
                self.mesh = trimesh.creation.box(extents=[10, 10, 3])
                self.mesh.apply_translation([5, 5, 1.5])
                self.mesh.visual.face_colors = [200, 200, 220, 255]
                return self.mesh
            
            # Create the base floor mesh using the valid faces
            floor_mesh = trimesh.Trimesh(vertices=v_3d, faces=valid_faces)
            
            # Create walls by extruding the edges
            # Get the edges that separate rooms from each other or from the outside
            edges = []
            for face in valid_faces:
                for i in range(3):
                    edge = (face[i], face[(i+1)%3])
                    edges.append(sorted(edge))
            
            # Count how many times each edge appears - boundary edges appear only once
            from collections import Counter
            edge_counts = Counter(map(tuple, edges))
            boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
            
            # Create wall meshes by extruding boundary edges
            wall_meshes = []
            for v1_idx, v2_idx in boundary_edges:
                # Skip edges with invalid indices
                if v1_idx > max_vertex_idx or v2_idx > max_vertex_idx:
                    continue
                    
                # Get the positions of the two vertices
                v1 = v_3d[v1_idx]
                v2 = v_3d[v2_idx]
                
                # Create wall vertices (4 corners: bottom and top pairs)
                wall_v = np.array([
                    v1,  # bottom left
                    v2,  # bottom right
                    [v2[0], v2[1], self.wall_height],  # top right
                    [v1[0], v1[1], self.wall_height]   # top left
                ])
                
                # Create wall faces (2 triangles)
                wall_f = np.array([
                    [0, 1, 2],  # First triangle
                    [0, 2, 3]   # Second triangle
                ])
                
                wall_mesh = trimesh.Trimesh(vertices=wall_v, faces=wall_f)
                wall_meshes.append(wall_mesh)
            
            # Convert the border map to a 3D mesh for floors
            floor_meshes = []
            for room_idx, room_name in self.room_name_dict.items():
                if room_idx == 0:  # Skip outside/walls
                    continue
                
                # Get mask for this room
                room_mask = (self.border_map == room_idx)
                
                # Skip empty rooms
                if not np.any(room_mask):
                    continue
                
                # Create a grid of 3D vertices for the floor
                h, w = self.border_map.shape
                grid_vertices = []
                grid_faces = []
                
                # For each cell in the room
                for i in range(h-1):
                    for j in range(w-1):
                        if not room_mask[i, j]:
                            continue
                        
                        # Add vertices for this cell (with z=0 for floor)
                        v_base = len(grid_vertices)
                        grid_vertices.extend([
                            [j, i, 0],
                            [j+1, i, 0],
                            [j+1, i+1, 0],
                            [j, i+1, 0]
                        ])
                        
                        # Add two triangles to form a square
                        grid_faces.extend([
                            [v_base, v_base+1, v_base+2],
                            [v_base, v_base+2, v_base+3]
                        ])
                
                if grid_vertices and grid_faces:
                    try:
                        floor_mesh = trimesh.Trimesh(
                            vertices=np.array(grid_vertices), 
                            faces=np.array(grid_faces)
                        )
                        floor_meshes.append(floor_mesh)
                    except Exception as e:
                        print(f"Warning: Could not create floor mesh for room {room_idx}: {e}")
            
            # Combine all meshes
            try:
                # Start with floor mesh
                meshes = [floor_mesh]
                
                # Add walls
                if wall_meshes:
                    meshes.extend(wall_meshes)
                
                # Add room floors
                if floor_meshes:
                    meshes.extend(floor_meshes)
                
                # Create combined mesh
                self.mesh = trimesh.util.concatenate(meshes)
                
                print(f"Generated house mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
                
                # Assign some default colors
                self.mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray for structure
                
                return self.mesh
            except Exception as e:
                print(f"Error combining meshes: {e}")
                traceback.print_exc()
                # Create a fallback simple box as the house
                self.mesh = trimesh.creation.box(extents=[10, 10, 3])
                self.mesh.apply_translation([5, 5, 1.5])
                self.mesh.visual.face_colors = [200, 200, 220, 255]
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
                
            # If mesh generation failed or we have an empty mesh, create a basic floor
            if self.mesh is None or len(self.mesh.vertices) < 3:
                print("Creating a basic floor mesh for furniture placement...")
                # Create a simple floor
                floor_size = 50
                self.mesh = trimesh.creation.box(extents=[floor_size, floor_size, 0.1])
                self.mesh.apply_translation([floor_size/2, floor_size/2, -0.05])
                self.mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray
            
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
                        
                        # Create furniture mesh based on the object type
                        if "bed" in obj_name.lower():
                            # Create a more detailed bed with base and mattress
                            base = trimesh.creation.box(extents=[size[0], size[1], size[2]/3])
                            mattress = trimesh.creation.box(extents=[size[0]*0.9, size[1]*0.9, size[2]/3])
                            mattress.apply_translation([size[0]*0.05, size[1]*0.05, size[2]/3])
                            
                            # Add headboard
                            headboard = trimesh.creation.box(extents=[size[0], size[1]*0.1, size[2]])
                            headboard.apply_translation([0, 0, 0])
                            
                            furniture = trimesh.util.concatenate([base, mattress, headboard])
                            furniture.visual.face_colors = [200, 100, 100, 255]  # Red for bed
                            
                        elif "table" in obj_name.lower() or "desk" in obj_name.lower():
                            # Table: top and legs
                            top_thickness = size[2] * 0.1
                            top = trimesh.creation.box(extents=[size[0], size[1], top_thickness])
                            top.apply_translation([0, 0, size[2] - top_thickness])
                            
                            leg_width = min(size[0], size[1]) * 0.1
                            leg_height = size[2] - top_thickness
                            
                            # Create four legs
                            leg1 = trimesh.creation.box(extents=[leg_width, leg_width, leg_height])
                            leg2 = trimesh.creation.box(extents=[leg_width, leg_width, leg_height])
                            leg3 = trimesh.creation.box(extents=[leg_width, leg_width, leg_height])
                            leg4 = trimesh.creation.box(extents=[leg_width, leg_width, leg_height])
                            
                            # Position legs at corners
                            leg1.apply_translation([leg_width/2, leg_width/2, 0])
                            leg2.apply_translation([size[0]-leg_width/2, leg_width/2, 0])
                            leg3.apply_translation([size[0]-leg_width/2, size[1]-leg_width/2, 0])
                            leg4.apply_translation([leg_width/2, size[1]-leg_width/2, 0])
                            
                            furniture = trimesh.util.concatenate([top, leg1, leg2, leg3, leg4])
                            furniture.visual.face_colors = [139, 69, 19, 255]  # Brown for tables
                            
                        elif "chair" in obj_name.lower():
                            # Chair: seat + backrest + legs
                            seat = trimesh.creation.box(extents=[size[0], size[1], size[2]/4])
                            seat.apply_translation([0, 0, size[2]/2])
                            
                            backrest = trimesh.creation.box(extents=[size[0], size[1]/8, size[2]/2])
                            backrest.apply_translation([0, 0, size[2]*3/4])
                            
                            # Create four legs
                            leg_width = min(size[0], size[1]) * 0.1
                            leg_height = size[2]/2
                            
                            leg1 = trimesh.creation.cylinder(radius=leg_width/2, height=leg_height)
                            leg2 = trimesh.creation.cylinder(radius=leg_width/2, height=leg_height)
                            leg3 = trimesh.creation.cylinder(radius=leg_width/2, height=leg_height)
                            leg4 = trimesh.creation.cylinder(radius=leg_width/2, height=leg_height)
                            
                            # Position legs at corners
                            leg1.apply_translation([leg_width, leg_width, leg_height/2])
                            leg2.apply_translation([size[0]-leg_width, leg_width, leg_height/2])
                            leg3.apply_translation([size[0]-leg_width, size[1]-leg_width, leg_height/2])
                            leg4.apply_translation([leg_width, size[1]-leg_width, leg_height/2])
                            
                            furniture = trimesh.util.concatenate([seat, backrest, leg1, leg2, leg3, leg4])
                            furniture.visual.face_colors = [169, 169, 169, 255]  # Gray for chairs
                            
                        elif "sofa" in obj_name.lower() or "couch" in obj_name.lower():
                            # Sofa: base + backrest + armrests
                            base = trimesh.creation.box(extents=[size[0], size[1], size[2]/2])
                            backrest = trimesh.creation.box(extents=[size[0], size[1]/4, size[2]])
                            backrest.apply_translation([0, 0, 0])
                            
                            # Add armrests
                            armrest1 = trimesh.creation.box(extents=[size[0]/8, size[1], size[2]*0.7])
                            armrest2 = trimesh.creation.box(extents=[size[0]/8, size[1], size[2]*0.7])
                            
                            armrest1.apply_translation([0, 0, size[2]/4])
                            armrest2.apply_translation([size[0]*7/8, 0, size[2]/4])
                            
                            furniture = trimesh.util.concatenate([base, backrest, armrest1, armrest2])
                            furniture.visual.face_colors = [106, 90, 205, 255]  # Purple for sofas
                            
                        elif "lamp" in obj_name.lower():
                            base = trimesh.creation.cylinder(radius=size[0]/4, height=size[2]*0.8)
                            shade = trimesh.creation.cylinder(radius=size[0]/2, height=size[2]*0.2)
                            shade.apply_translation([0, 0, size[2]*0.7])
                            furniture = trimesh.util.concatenate([base, shade])
                            furniture.visual.face_colors = [255, 215, 0, 255]  # Gold for lamps
                            
                        elif "cabinet" in obj_name.lower() or "wardrobe" in obj_name.lower():
                            # Cabinet: box with details
                            main_body = trimesh.creation.box(extents=size)
                            
                            # Add doors/drawers
                            num_doors = max(1, int(size[0] / 0.6))
                            door_width = size[0] / num_doors
                            
                            doors = []
                            for i in range(num_doors):
                                door = trimesh.creation.box(extents=[door_width*0.9, size[1]*0.05, size[2]*0.9])
                                door.apply_translation([i*door_width + door_width*0.05, size[1]*0.95, size[2]*0.05])
                                doors.append(door)
                            
                            if doors:
                                all_doors = trimesh.util.concatenate(doors)
                                furniture = trimesh.util.concatenate([main_body, all_doors])
                            else:
                                furniture = main_body
                                
                            furniture.visual.face_colors = [160, 82, 45, 255]  # Brown for cabinets
                            
                        else:
                            # Default: box-shaped furniture with random color
                            furniture = trimesh.creation.box(extents=size)
                            # Generate a random color with good visibility
                            rng = np.random.RandomState(hash(obj_name) % 2**32)
                            color = rng.randint(100, 240, size=3).tolist() + [255]
                            furniture.visual.face_colors = color
                        
                        # Apply rotation (around z-axis)
                        rotation = trimesh.transformations.rotation_matrix(
                            angle=np.radians(float(ang)),
                            direction=[0, 0, 1],
                            point=[0, 0, 0]
                        )
                        furniture.apply_transform(rotation)
                        
                        # Apply translation (position)
                        # Convert 2D position to 3D (add z=0 for placing on floor)
                        translation = np.array([pos[0], pos[1], 0])
                        furniture.apply_translation(translation)
                        
                        # Add this furniture to the list
                        furniture_meshes.append(furniture)
                        
                    except Exception as e:
                        print(f"Error adding furniture {obj_name} in room {room_idx}: {e}")
                        traceback.print_exc()
            
            # If we have furniture, combine with the house mesh
            if furniture_meshes:
                try:
                    # Add the house mesh
                    meshes_to_combine = [self.mesh] + furniture_meshes
                    
                    # Create the combined mesh
                    combined_mesh = trimesh.util.concatenate(meshes_to_combine)
                    self.mesh = combined_mesh
                    
                    print(f"Added {len(furniture_meshes)} furniture objects to the house mesh")
                except Exception as e:
                    print(f"Error combining furniture with house: {e}")
                    traceback.print_exc()
                    
                    # Try to at least return the furniture
                    try:
                        if len(furniture_meshes) > 0:
                            self.mesh = trimesh.util.concatenate(furniture_meshes)
                            print("Created mesh with furniture only (house mesh excluded due to error)")
                    except Exception as e2:
                        print(f"Error creating furniture-only mesh: {e2}")
                        # Create a fallback mesh
                        self.mesh = trimesh.creation.box(extents=[10, 10, 3])
                        self.mesh.apply_translation([5, 5, 1.5])
                        self.mesh.visual.face_colors = [200, 200, 220, 255]
            
            # Return the combined mesh
            return self.mesh
            
        except Exception as e:
            print(f"Error adding furniture to mesh: {e}")
            traceback.print_exc()
            
            # If the mesh is already valid, return it
            if self.mesh is not None:
                return self.mesh
                
            # Otherwise create a simple fallback
            self.mesh = trimesh.creation.box(extents=[10, 10, 3])
            self.mesh.apply_translation([5, 5, 1.5])
            self.mesh.visual.face_colors = [200, 200, 220, 255]
            return self.mesh
    
    def export_mesh(self, filename="output/3d_model.obj"):
        """Export the house mesh to an OBJ file"""
        if self.mesh is None:
            self.mesh = self.generate_house_mesh()
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Export the mesh
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