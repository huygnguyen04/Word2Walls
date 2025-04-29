import os
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import traceback

class SceneRenderer:
    def __init__(self, mesh=None):
        self.mesh = mesh
        self.output_dir = "output/3d_renders"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default settings
        self.light_intensity = 3.0
        self.ambient_intensity = 0.5
        self.bg_color = [0.9, 0.9, 0.9, 1.0]  # Light gray background
        
        print(f"SceneRenderer initialized. Output directory: {self.output_dir}")
        
    def set_mesh(self, mesh):
        """Set the mesh to be rendered"""
        self.mesh = mesh
        
    def render_simple_scene(self, output_path=None):
        """Render a simple scene with basic lighting and default camera"""
        try:
            print("Rendering simple scene...")
            
            # Default output path
            if output_path is None:
                output_path = os.path.join(self.output_dir, "simple_render.png")
                
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check if mesh is valid
            if self.mesh is None:
                print("Warning: No mesh set for rendering. Creating a default scene.")
                # Create a simple scene with example furniture
                self._render_default_scene(output_path)
                return output_path
                
            # Convert trimesh to pyrender mesh
            try:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.85, 0.85, 0.85, 1.0],
                    metallicFactor=0.1,
                    roughnessFactor=0.8
                )
                
                # Convert trimesh to pyrender mesh
                render_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material, smooth=False)
                
                # Create a scene
                scene = pyrender.Scene(ambient_light=self.ambient_intensity * np.array([1.0, 1.0, 1.0, 1.0]),
                                      bg_color=self.bg_color)
                
                # Add the mesh to the scene
                scene.add(render_mesh)
                
                # Add a camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0/3.0, znear=0.1, zfar=100.0)
                
                # Calculate scene bounds for camera placement
                bounds = self.mesh.bounds
                center = (bounds[0] + bounds[1]) / 2.0
                extents = bounds[1] - bounds[0]
                size = np.max(extents)
                
                # Place camera based on the scene size
                camera_pose = np.array([
                    [1.0, 0.0, 0.0, center[0]],
                    [0.0, 0.0, -1.0, center[1] - 1.5 * size],
                    [0.0, 1.0, 0.0, center[2] + 0.5 * size],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                scene.add(camera, pose=camera_pose)
                
                # Add lights
                # Directional light from above
                direc_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
                direc_light_pose = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                scene.add(direc_light, pose=direc_light_pose)
                
                # Point light near the camera
                point_light = pyrender.PointLight(color=np.ones(3), intensity=self.light_intensity/2)
                scene.add(point_light, pose=camera_pose)
                
                # Render the scene
                r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
                color, depth = r.render(scene)
                r.delete()
                
                # Save the rendered image
                plt.figure(figsize=(10, 7.5))
                plt.imshow(color)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Simple render saved to {output_path}")
                return output_path
                
            except Exception as e:
                print(f"Error during rendering process: {e}")
                traceback.print_exc()
                # Fall back to default scene
                self._render_default_scene(output_path)
                return output_path
                
        except Exception as e:
            print(f"Error in render_simple_scene: {e}")
            traceback.print_exc()
            return None
            
    def _render_default_scene(self, output_path):
        """Render a default scene with sample furniture when no mesh is available"""
        try:
            # Create a simple room
            floor = trimesh.creation.box(extents=[10, 10, 0.1])
            floor.apply_translation([5, 5, -0.05])
            floor.visual.face_colors = [200, 200, 200, 255]
            
            wall1 = trimesh.creation.box(extents=[10, 0.1, 3])
            wall1.apply_translation([5, 0, 1.5])
            wall1.visual.face_colors = [220, 220, 220, 255]
            
            wall2 = trimesh.creation.box(extents=[0.1, 10, 3])
            wall2.apply_translation([0, 5, 1.5])
            wall2.visual.face_colors = [220, 220, 220, 255]
            
            # Add some furniture
            bed = trimesh.creation.box(extents=[2.5, 4, 0.5])
            bed.apply_translation([7.5, 2.5, 0.25])
            bed.visual.face_colors = [200, 100, 100, 255]
            
            nightstand = trimesh.creation.box(extents=[1, 1, 0.8])
            nightstand.apply_translation([6, 1.5, 0.4])
            nightstand.visual.face_colors = [139, 69, 19, 255]
            
            # Lamp (cylinder + sphere)
            lamp_base = trimesh.creation.cylinder(radius=0.2, height=0.6)
            lamp_base.apply_translation([6, 1.5, 0.8])
            lamp_base.visual.face_colors = [100, 100, 100, 255]
            
            lamp_shade = trimesh.creation.cylinder(radius=0.3, height=0.3)
            lamp_shade.apply_translation([6, 1.5, 1.25])
            lamp_shade.visual.face_colors = [255, 215, 0, 255]
            
            wardrobe = trimesh.creation.box(extents=[1.5, 0.6, 2.0])
            wardrobe.apply_translation([3, 9, 1.0])
            wardrobe.visual.face_colors = [120, 81, 45, 255]
            
            # Combine all meshes
            room_mesh = trimesh.util.concatenate([floor, wall1, wall2, bed, nightstand, lamp_base, lamp_shade, wardrobe])
            
            # Temporarily set this as the current mesh
            temp_mesh = self.mesh
            self.mesh = room_mesh
            
            # Render the scene
            result = self.render_simple_scene(output_path)
            
            # Restore the original mesh
            self.mesh = temp_mesh
            
            return result
            
        except Exception as e:
            print(f"Error creating default scene: {e}")
            traceback.print_exc()
            
            # Create a very basic image with text
            img = Image.new('RGB', (800, 600), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            
            # Draw text explaining the error
            try:
                # Try to get a font
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                d.text((50, 50), "Could not render 3D scene", fill=(0, 0, 0), font=font)
                d.text((50, 80), f"Error: {str(e)}", fill=(200, 0, 0), font=font)
                d.text((50, 110), "Please check that your 3D libraries are installed correctly.", fill=(0, 0, 0), font=font)
                
                # Save the image
                img.save(output_path)
                return output_path
                
            except Exception as inner_e:
                print(f"Failed to create even a basic error image: {inner_e}")
                return None
    
    def render_multiple_views(self, output_prefix=None, combined=True):
        """Render the scene from multiple views"""
        try:
            print("Rendering multiple views...")
            
            if self.mesh is None:
                print("Warning: No mesh available for rendering views")
                return self.render_simple_scene()
                
            if output_prefix is None:
                output_prefix = os.path.join(self.output_dir, "view")
                
            # Views to render
            views = {
                "top": {
                    "rotation": [0, 0, 0],
                    "distance_factor": 1.5,
                    "elevation": 90,
                    "filename": f"{output_prefix}_top.png"
                },
                "perspective": {
                    "rotation": [30, -45, 0],  # Looking from top-front-right
                    "distance_factor": 1.5,
                    "elevation": 30,
                    "filename": f"{output_prefix}_perspective.png"
                },
                "front": {
                    "rotation": [0, 0, 0],
                    "distance_factor": 1.2,
                    "elevation": 0,
                    "filename": f"{output_prefix}_front.png"
                },
                "side": {
                    "rotation": [0, -90, 0],
                    "distance_factor": 1.2,
                    "elevation": 0,
                    "filename": f"{output_prefix}_side.png"
                }
            }
            
            # Calculate scene bounds for camera placement
            bounds = self.mesh.bounds
            center = (bounds[0] + bounds[1]) / 2.0
            extents = bounds[1] - bounds[0]
            size = np.max(extents)
            
            # Render each view
            rendered_paths = []
            
            for view_name, view_params in views.items():
                try:
                    # Convert trimesh to pyrender mesh
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.85, 0.85, 0.85, 1.0],
                        metallicFactor=0.1,
                        roughnessFactor=0.8
                    )
                    render_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material, smooth=False)
                    
                    # Create scene
                    scene = pyrender.Scene(ambient_light=self.ambient_intensity * np.array([1.0, 1.0, 1.0, 1.0]),
                                          bg_color=self.bg_color)
                    scene.add(render_mesh)
                    
                    # Create camera
                    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0/3.0, znear=0.1, zfar=100.0)
                    
                    # Position and orient camera
                    rotation = view_params["rotation"]
                    distance = view_params["distance_factor"] * size
                    elevation = view_params["elevation"]
                    
                    # Calculate camera position
                    if elevation == 90:  # Top view
                        camera_pos = np.array([center[0], center[1], center[2] + distance])
                        camera_pose = np.array([
                            [1.0, 0.0, 0.0, camera_pos[0]],
                            [0.0, 1.0, 0.0, camera_pos[1]],
                            [0.0, 0.0, 1.0, camera_pos[2]],
                            [0.0, 0.0, 0.0, 1.0]
                        ])
                    elif elevation == 0:  # Side view
                        if "front" in view_name:
                            camera_pos = np.array([center[0], center[1] - distance, center[2]])
                            camera_pose = np.array([
                                [1.0, 0.0, 0.0, camera_pos[0]],
                                [0.0, 0.0, -1.0, camera_pos[1]],
                                [0.0, 1.0, 0.0, camera_pos[2]],
                                [0.0, 0.0, 0.0, 1.0]
                            ])
                        else:  # side view
                            camera_pos = np.array([center[0] - distance, center[1], center[2]])
                            camera_pose = np.array([
                                [0.0, -1.0, 0.0, camera_pos[0]],
                                [1.0, 0.0, 0.0, camera_pos[1]],
                                [0.0, 0.0, 1.0, camera_pos[2]],
                                [0.0, 0.0, 0.0, 1.0]
                            ])
                    else:  # Perspective view
                        x_angle = np.radians(rotation[0])
                        y_angle = np.radians(rotation[1])
                        
                        # Calculate position
                        x = center[0] + distance * np.sin(y_angle)
                        y = center[1] + distance * np.cos(y_angle) * np.cos(x_angle)
                        z = center[2] + distance * np.sin(x_angle)
                        
                        # Look at center
                        camera_pos = np.array([x, y, z])
                        camera_target = center
                        camera_up = np.array([0, 0, 1])  # Z-up
                        
                        camera_pose = self._look_at(camera_pos, camera_target, camera_up)
                    
                    # Add camera to scene
                    scene.add(camera, pose=camera_pose)
                    
                    # Add lights
                    # Main directional light
                    direc_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
                    direc_light_pose = np.eye(4)
                    direc_light_pose[:3, 3] = camera_pos + np.array([0, 0, size/2])
                    scene.add(direc_light, pose=direc_light_pose)
                    
                    # Fill light from opposite direction
                    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity * 0.5)
                    fill_light_pose = np.eye(4)
                    fill_light_pose[:3, 3] = camera_pos * -0.7 + np.array([0, 0, size])
                    scene.add(fill_light, pose=fill_light_pose)
                    
                    # Add point light near the camera
                    point_light = pyrender.PointLight(color=np.ones(3), intensity=self.light_intensity)
                    point_light_pose = np.eye(4)
                    point_light_pose[:3, 3] = camera_pos
                    scene.add(point_light, pose=point_light_pose)
                    
                    # Render the scene
                    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
                    color, depth = r.render(scene)
                    r.delete()
                    
                    # Add view name as a caption
                    img = Image.fromarray(color)
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((20, 20), view_name.upper(), fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
                    color = np.array(img)
                    
                    # Save the rendered image
                    plt.figure(figsize=(10, 7.5))
                    plt.imshow(color)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(view_params["filename"], dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    rendered_paths.append(view_params["filename"])
                    print(f"Rendered {view_name} view: {view_params['filename']}")
                    
                except Exception as e:
                    print(f"Error rendering {view_name} view: {e}")
                    traceback.print_exc()
            
            # Create a combined view if requested and if we have renders
            if combined and rendered_paths:
                try:
                    combined_path = f"{output_prefix}_combined.png"
                    
                    # Load rendered images
                    imgs = [Image.open(path) for path in rendered_paths if os.path.exists(path)]
                    if not imgs:
                        return rendered_paths
                    
                    # Get dimensions
                    img_width = imgs[0].width
                    img_height = imgs[0].height
                    
                    # Create a 2x2 grid
                    combined_img = Image.new('RGB', (img_width * 2, img_height * 2), (240, 240, 240))
                    
                    # Paste images into the grid
                    for i, img in enumerate(imgs[:4]):  # Use up to 4 images
                        x = (i % 2) * img_width
                        y = (i // 2) * img_height
                        combined_img.paste(img, (x, y))
                    
                    # Save combined image
                    combined_img.save(combined_path)
                    print(f"Created combined view: {combined_path}")
                    
                    # Add the combined path to the return list
                    rendered_paths.append(combined_path)
                    
                except Exception as e:
                    print(f"Error creating combined view: {e}")
                    traceback.print_exc()
            
            return rendered_paths
                
        except Exception as e:
            print(f"Error in render_multiple_views: {e}")
            traceback.print_exc()
            return [self.render_simple_scene()]
    
    def _look_at(self, eye, target, up):
        """Helper function to create a transformation matrix for a camera looking at a target"""
        forward = np.array(target) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        
        side = np.cross(forward, up)
        side = side / np.linalg.norm(side)
        
        new_up = np.cross(side, forward)
        
        rotation = np.eye(4)
        rotation[:3, 0] = side
        rotation[:3, 1] = new_up
        rotation[:3, 2] = -forward
        
        translation = np.eye(4)
        translation[:3, 3] = -np.array(eye)
        
        return np.matmul(rotation, translation)
    
    def render_turntable_animation(self, num_frames=20, output_path=None):
        """Create a turntable animation around the object"""
        try:
            print("Creating turntable animation...")
            
            if self.mesh is None:
                print("Warning: No mesh available for rendering animation")
                return self.render_simple_scene()
                
            if output_path is None:
                output_path = os.path.join(self.output_dir, "turntable_animation.gif")
                
            # Calculate scene bounds for camera placement
            bounds = self.mesh.bounds
            center = (bounds[0] + bounds[1]) / 2.0
            extents = bounds[1] - bounds[0]
            size = np.max(extents)
            
            # Distance for the camera
            distance = size * 1.5
            
            # Create frames
            frames = []
            
            for i in range(num_frames):
                # Calculate camera angle around the object
                angle = i * (2 * np.pi / num_frames)
                
                # Calculate camera position
                x = center[0] + distance * np.sin(angle)
                y = center[1] + distance * np.cos(angle)
                z = center[2] + size * 0.5  # Slightly above center
                
                # Create and set up the scene
                # Convert trimesh to pyrender mesh
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.85, 0.85, 0.85, 1.0],
                    metallicFactor=0.1,
                    roughnessFactor=0.8
                )
                render_mesh = pyrender.Mesh.from_trimesh(self.mesh, material=material, smooth=False)
                
                # Create scene
                scene = pyrender.Scene(ambient_light=self.ambient_intensity * np.array([1.0, 1.0, 1.0, 1.0]),
                                      bg_color=self.bg_color)
                scene.add(render_mesh)
                
                # Create camera and look at the center
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0/3.0, znear=0.1, zfar=100.0)
                camera_pos = np.array([x, y, z])
                camera_target = center
                camera_up = np.array([0, 0, 1])  # Z-up
                camera_pose = self._look_at(camera_pos, camera_target, camera_up)
                scene.add(camera, pose=camera_pose)
                
                # Add lights
                # Main directional light
                direc_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
                direc_light_pose = np.eye(4)
                direc_light_pose[:3, 3] = camera_pos + np.array([0, 0, size/2])
                scene.add(direc_light, pose=direc_light_pose)
                
                # Fill light from opposite direction
                fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity * 0.5)
                fill_light_pose = np.eye(4)
                fill_light_pose[:3, 3] = camera_pos * -0.7 + np.array([0, 0, size])
                scene.add(fill_light, pose=fill_light_pose)
                
                # Render the scene
                r = pyrender.OffscreenRenderer(viewport_width=600, viewport_height=600)
                color, depth = r.render(scene)
                r.delete()
                
                # Convert to PIL image for GIF
                frames.append(Image.fromarray(color))
                
                # Update progress
                if i % 5 == 0:
                    print(f"Rendered frame {i+1}/{num_frames}")
            
            # Save as animated GIF
            if frames:
                frames[0].save(
                    output_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=100,  # milliseconds per frame
                    loop=0  # 0 means loop forever
                )
                print(f"Turntable animation saved to {output_path}")
                return output_path
            else:
                return None
                
        except Exception as e:
            print(f"Error creating turntable animation: {e}")
            traceback.print_exc()
            return None 