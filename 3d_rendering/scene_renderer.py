import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
import os

class SceneRenderer:
    def __init__(self, mesh=None):
        """
        Initialize the scene renderer.
        
        Args:
            mesh: Trimesh object to render
        """
        self.mesh = mesh
        self.scene = None
        self.camera = None
        self.render_dir = "output/3d_renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
    def setup_scene(self, mesh=None):
        """Set up the rendering scene with lights and camera"""
        if mesh is not None:
            self.mesh = mesh
            
        if self.mesh is None:
            raise ValueError("No mesh provided for rendering")
            
        # Create pyrender scene
        self.scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
        
        # Add mesh to scene
        if isinstance(self.mesh, list):
            # Handle multiple meshes
            for m in self.mesh:
                self.scene.add(pyrender.Mesh.from_trimesh(m))
        else:
            # Single mesh
            self.scene.add(pyrender.Mesh.from_trimesh(self.mesh))
        
        # Add lights
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=np.eye(4))
        
        # Add a point light at the center of the scene
        center = self.mesh.centroid
        light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = center + np.array([0, 0, 2.0])  # Light above center
        self.scene.add(light, pose=light_pose)
        
        # Default camera setup
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        self.camera = camera
        
        return self.scene
    
    def render_from_trajectory(self, trajectory, output_prefix="view"):
        """Render views from a given camera trajectory"""
        if self.scene is None:
            self.setup_scene()
            
        renders = []
        
        for i, position in enumerate(trajectory):
            # Calculate camera position and target
            camera_pos = position
            
            # Look at the center of the scene
            center = self.mesh.centroid
            
            # Create camera transformation matrix (look-at)
            z = (center - camera_pos)
            z = z / np.linalg.norm(z)
            x = np.cross(np.array([0, 0, 1]), z)
            if np.allclose(x, 0):
                x = np.array([1, 0, 0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            
            camera_pose = np.eye(4)
            camera_pose[:3, 0] = x
            camera_pose[:3, 1] = y
            camera_pose[:3, 2] = z
            camera_pose[:3, 3] = camera_pos
            
            # Add camera to scene
            if self.camera in self.scene.nodes:
                self.scene.remove_node(self.scene.get_nodes(obj=self.camera)[0])
            camera_node = self.scene.add(self.camera, pose=camera_pose)
            
            # Render
            r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
            color, depth = r.render(self.scene)
            
            # Save the render
            image_path = os.path.join(self.render_dir, f"{output_prefix}_{i:03d}.png")
            plt.figure(figsize=(8, 6))
            plt.imshow(color)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            
            renders.append(color)
            
        return renders
    
    def render_orthographic_views(self):
        """Render orthographic top, front, and side views of the scene"""
        if self.scene is None:
            self.setup_scene()
            
        # Calculate scene bounds
        bounds = self.mesh.bounds
        extents = bounds[1] - bounds[0]
        scale = np.max(extents) * 1.2  # Add some margin
        center = (bounds[0] + bounds[1]) / 2.0
        
        # Create orthographic camera
        camera = pyrender.OrthographicCamera(xmag=scale/2, ymag=scale/2)
        
        # Define views
        views = {
            'top': {
                'position': center + np.array([0, 0, scale]),
                'target': center,
                'up': np.array([0, 1, 0])
            },
            'front': {
                'position': center + np.array([0, -scale, 0]),
                'target': center,
                'up': np.array([0, 0, 1])
            },
            'side': {
                'position': center + np.array([scale, 0, 0]),
                'target': center,
                'up': np.array([0, 0, 1])
            }
        }
        
        renders = {}
        
        for view_name, view_params in views.items():
            # Create look-at matrix
            position = view_params['position']
            target = view_params['target']
            up = view_params['up']
            
            z = target - position
            z = z / np.linalg.norm(z)
            x = np.cross(up, z)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            
            camera_pose = np.eye(4)
            camera_pose[:3, 0] = x
            camera_pose[:3, 1] = y
            camera_pose[:3, 2] = z
            camera_pose[:3, 3] = position
            
            # Remove existing camera and add new one
            if camera in self.scene.nodes:
                self.scene.remove_node(self.scene.get_nodes(obj=camera)[0])
            self.scene.add(camera, pose=camera_pose)
            
            # Render
            r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=640)
            color, depth = r.render(self.scene)
            
            # Save render
            image_path = os.path.join(self.render_dir, f"{view_name}_view.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(color)
            plt.axis('off')
            plt.title(f"{view_name.capitalize()} View")
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            
            renders[view_name] = color
            
        return renders
    
    def create_flythrough_animation(self, trajectory, output_path="output/3d_renders/flythrough.gif", fps=10):
        """Create a GIF animation of a camera flythrough along trajectory"""
        if self.scene is None:
            self.setup_scene()
            
        # Render all frames
        frames = self.render_from_trajectory(trajectory, output_prefix="flythrough")
        
        # Convert to PIL images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=1000//fps,
            loop=0
        )
        
        return output_path
    
    def render_360_turntable(self, num_frames=36, output_path="output/3d_renders/turntable.gif", fps=10):
        """Create a 360-degree turntable animation around the model"""
        if self.scene is None:
            self.setup_scene()
            
        # Calculate scene center and radius
        center = self.mesh.centroid
        radius = np.max(np.linalg.norm(self.mesh.vertices - center, axis=1)) * 1.5
        
        # Create circular trajectory around the model
        angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        positions = []
        
        for angle in angles:
            # Position camera in a circle around the model
            x = center[0] + radius * np.sin(angle)
            y = center[1] + radius * np.cos(angle)
            z = center[2] + radius * 0.3  # Slightly above center
            
            positions.append(np.array([x, y, z]))
        
        # Render the frames
        frames = []
        for i, position in enumerate(positions):
            # Calculate camera transformation
            target = center
            
            # Create look-at matrix
            z = target - position
            z = z / np.linalg.norm(z)
            x = np.cross(np.array([0, 0, 1]), z)
            if np.allclose(x, 0):
                x = np.array([1, 0, 0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            
            camera_pose = np.eye(4)
            camera_pose[:3, 0] = x
            camera_pose[:3, 1] = y
            camera_pose[:3, 2] = z
            camera_pose[:3, 3] = position
            
            # Add camera to scene
            if self.camera in self.scene.nodes:
                self.scene.remove_node(self.scene.get_nodes(obj=self.camera)[0])
            camera_node = self.scene.add(self.camera, pose=camera_pose)
            
            # Render
            r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
            color, depth = r.render(self.scene)
            
            # Save the frame
            image_path = os.path.join(self.render_dir, f"turntable_{i:03d}.png")
            plt.imsave(image_path, color)
            
            frames.append(Image.fromarray(color))
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=1000//fps,
            loop=0
        )
        
        return output_path 