import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, disk

class TrajectoryGenerator:
    def __init__(self, border_map, objects_pos=None, objects_size=None):
        """
        Initialize the trajectory generator.
        
        Args:
            border_map: The border map with room indices
            objects_pos: Dictionary of object positions {obj_name: (x, y)}
            objects_size: Dictionary of object sizes {obj_name: (width, height)}
        """
        self.border_map = border_map
        self.objects_pos = objects_pos if objects_pos is not None else {}
        self.objects_size = objects_size if objects_size is not None else {}
        self.trajectory = None
        
    def create_obstacle_field(self):
        """Create an obstacle field based on the border map and object positions"""
        obstacle_map = np.zeros_like(self.border_map, dtype=np.float32)
        
        # Walls are obstacles (where border_map == 0)
        obstacle_map[self.border_map == 0] = 1.0
        
        # Add objects as obstacles
        for obj_name, pos in self.objects_pos.items():
            if obj_name in self.objects_size:
                x, y = pos
                w, h = self.objects_size[obj_name]
                obstacle_map[x:x+w, y:y+h] = 0.8  # Objects have lower obstacle value than walls
        
        # Create distance field from obstacles
        distance_field = np.zeros_like(obstacle_map, dtype=np.float32)
        for i in range(1, 20):  # Limit the distance calculation to 20 pixels
            mask = binary_dilation(obstacle_map > 0, disk(i)) & ~binary_dilation(obstacle_map > 0, disk(i-1))
            distance_field[mask] = 1.0 - (i / 20)
            
        return distance_field
    
    def extract_keypoints(self, obstacle_field, n_points=20):
        """Extract keypoints for trajectory from the obstacle field"""
        # Find non-obstacle points
        free_space = (obstacle_field == 0)
        if not np.any(free_space):
            # If no free space, just sample random points
            return np.random.rand(n_points, 2) * np.array(obstacle_field.shape)
        
        # Find points close to walls but not obstacles
        room_points = np.argwhere(free_space)
        
        # If we have too few points, duplicate them
        if len(room_points) < n_points:
            indices = np.random.choice(len(room_points), n_points, replace=True)
        else:
            indices = np.random.choice(len(room_points), n_points, replace=False)
            
        keypoints = room_points[indices]
        
        # Add object-centric views if we have objects
        if self.objects_pos:
            object_centers = []
            for obj_name, pos in self.objects_pos.items():
                if obj_name in self.objects_size:
                    x, y = pos
                    w, h = self.objects_size[obj_name]
                    center = np.array([x + w//2, y + h//2])
                    object_centers.append(center)
            
            if object_centers:
                object_centers = np.vstack(object_centers)
                # Replace some keypoints with object centers
                n_objects = min(len(object_centers), n_points // 4)
                obj_indices = np.random.choice(len(object_centers), n_objects, replace=False)
                keypoints[-n_objects:] = object_centers[obj_indices]
        
        return keypoints
    
    def generate_trajectory(self):
        """Generate a camera trajectory through the house"""
        # Create obstacle field
        obstacle_field = self.create_obstacle_field()
        
        # Extract keypoints
        keypoints = self.extract_keypoints(obstacle_field)
        
        # Order keypoints to create a reasonable path (approximate TSP)
        ordered_points = [keypoints[0]]
        remaining = list(range(1, len(keypoints)))
        
        while remaining:
            last = ordered_points[-1]
            distances = [np.sum((last - keypoints[i])**2) for i in remaining]
            closest_idx = remaining[np.argmin(distances)]
            ordered_points.append(keypoints[closest_idx])
            remaining.remove(closest_idx)
        
        # Add height information (fixed camera height)
        trajectory_3d = np.zeros((len(ordered_points), 3))
        trajectory_3d[:, :2] = ordered_points
        trajectory_3d[:, 2] = 1.6  # Fixed camera height
        
        self.trajectory = trajectory_3d
        return trajectory_3d
    
    def visualize_trajectory(self, save_path=None):
        """Visualize the trajectory on the border map"""
        if self.trajectory is None:
            self.generate_trajectory()
            
        plt.figure(figsize=(10, 10))
        plt.imshow(self.border_map, cmap='tab20b')
        
        # Plot trajectory
        plt.plot(self.trajectory[:, 1], self.trajectory[:, 0], 'r-', linewidth=2)
        plt.plot(self.trajectory[:, 1], self.trajectory[:, 0], 'ro', markersize=5)
        
        # Plot objects if available
        for obj_name, pos in self.objects_pos.items():
            if obj_name in self.objects_size:
                x, y = pos
                w, h = self.objects_size[obj_name]
                plt.gca().add_patch(plt.Rectangle((y, x), h, w, fill=False, edgecolor='green', linewidth=2))
        
        plt.title('Camera Trajectory')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 