import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class SDSOptimizer:
    def __init__(self, objects_pos, objects_size, objects_ang, room_mask):
        """
        Initialize the SDS optimizer for refining object positions.
        
        Args:
            objects_pos: Dictionary of object positions {obj_name: (x, y)}
            objects_size: Dictionary of object sizes {obj_name: (width, height)}
            objects_ang: Dictionary of object angles {obj_name: angle}
            room_mask: Binary mask of the room (1 for room, 0 for walls)
        """
        self.objects_pos = objects_pos
        self.objects_size = objects_size
        self.objects_ang = objects_ang
        self.room_mask = room_mask
        
        # Convert to tensors for optimization
        self.object_names = list(objects_pos.keys())
        
        # Only initialize pos_tensor if we have objects
        if self.object_names:
            self.pos_tensor = torch.tensor(
                [[objects_pos[name][0], objects_pos[name][1]] 
                 for name in self.object_names], 
                dtype=torch.float32,
                requires_grad=True
            )
        else:
            self.pos_tensor = None
        
        # Define loss weights
        self.w_boundary = 10.0  # Stay within room boundary
        self.w_overlap = 5.0    # Avoid object overlap
        self.w_wall = 8.0       # Align with walls
        self.w_spacing = 3.0    # Maintain spacing between objects
        
    def _room_boundary_loss(self, pos):
        """Loss to keep objects within room boundaries"""
        loss = torch.tensor(0.0, dtype=torch.float32, device=pos.device, requires_grad=True)
        mask_tensor = torch.from_numpy(self.room_mask).float()
        
        for i, obj_name in enumerate(self.object_names):
            w, h = self.objects_size[obj_name]
            x, y = pos[i, 0], pos[i, 1]
            
            # Check corners of the object
            corners = [
                (x, y),
                (x + w, y),
                (x, y + h),
                (x + w, y + h)
            ]
            
            for cx, cy in corners:
                cx_int, cy_int = int(cx.item()), int(cy.item())
                # Ensure within array bounds
                if 0 <= cx_int < mask_tensor.shape[0] and 0 <= cy_int < mask_tensor.shape[1]:
                    # Add loss if corner is outside room (mask == 0)
                    if mask_tensor[cx_int, cy_int] == 0:
                        # Use differentiable operation
                        penalty = torch.tensor(1.0, dtype=torch.float32, device=pos.device)
                        loss = loss + penalty
                else:
                    # Corner is outside the entire mask
                    penalty = torch.tensor(1.0, dtype=torch.float32, device=pos.device)
                    loss = loss + penalty
                    
        return loss
    
    def _overlap_loss(self, pos):
        """Loss to prevent objects from overlapping"""
        loss = torch.tensor(0.0, dtype=torch.float32, device=pos.device)
        
        for i, obj_i in enumerate(self.object_names):
            w_i, h_i = self.objects_size[obj_i]
            x_i, y_i = pos[i, 0], pos[i, 1]
            
            for j, obj_j in enumerate(self.object_names):
                if i == j:
                    continue
                    
                w_j, h_j = self.objects_size[obj_j]
                x_j, y_j = pos[j, 0], pos[j, 1]
                
                # Check for overlap using rectangular intersection
                overlap_x = torch.max(torch.tensor(0.0, device=pos.device), 
                                   torch.min(x_i + w_i, x_j + w_j) - 
                                   torch.max(x_i, x_j))
                
                overlap_y = torch.max(torch.tensor(0.0, device=pos.device), 
                                   torch.min(y_i + h_i, y_j + h_j) - 
                                   torch.max(y_i, y_j))
                
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0:
                    loss = loss + overlap_area / (w_i * h_i + w_j * h_j - overlap_area)
                    
        return loss
    
    def _wall_alignment_loss(self, pos):
        """Loss to encourage alignment with walls"""
        loss = torch.tensor(0.0, dtype=torch.float32, device=pos.device)
        mask_tensor = torch.from_numpy(self.room_mask).float()
        
        # Find wall pixels (transitions from 0 to 1 in the mask)
        wall_h = (mask_tensor[:, 1:] != mask_tensor[:, :-1]).float()
        wall_v = (mask_tensor[1:, :] != mask_tensor[:-1, :]).float()
        
        for i, obj_name in enumerate(self.object_names):
            w, h = self.objects_size[obj_name]
            x, y = pos[i, 0], pos[i, 1]
            
            # Only consider objects that should be wall-aligned (based on original angle)
            ang = self.objects_ang.get(obj_name, "N")
            if ang not in ["N", "S", "E", "W"]:
                continue
                
            min_wall_dist = float('inf')
            
            # Check horizontal and vertical wall alignment
            for wx in range(mask_tensor.shape[0] - 1):
                for wy in range(mask_tensor.shape[1] - 1):
                    # Check only at wall locations
                    if wall_h[wx, wy] == 1 or wall_v[wx, wy] == 1:
                        # Distance from object edges to this wall pixel
                        dist_left = abs(float(x.item()) - wx)
                        dist_right = abs(float(x.item() + w) - wx)
                        dist_top = abs(float(y.item()) - wy)
                        dist_bottom = abs(float(y.item() + h) - wy)
                        
                        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                        min_wall_dist = min(min_wall_dist, min_dist)
            
            # Penalize if not aligned with any wall
            if min_wall_dist > 5:  # Threshold for wall alignment
                loss = loss + torch.tensor(1.0, dtype=torch.float32, device=pos.device)
                
        return loss
    
    def _spacing_loss(self, pos):
        """Loss to maintain good spacing between objects"""
        loss = torch.tensor(0.0, dtype=torch.float32, device=pos.device)
        min_spacing = 3.0  # Minimum desired spacing
        
        for i, obj_i in enumerate(self.object_names):
            w_i, h_i = self.objects_size[obj_i]
            x_i, y_i = pos[i, 0], pos[i, 1]
            
            for j, obj_j in enumerate(self.object_names):
                if i == j:
                    continue
                    
                w_j, h_j = self.objects_size[obj_j]
                x_j, y_j = pos[j, 0], pos[j, 1]
                
                # Calculate minimum distance between objects
                dist_x = torch.max(torch.tensor(0.0, device=pos.device), 
                                  torch.min(x_i + w_i + min_spacing, x_j + w_j + min_spacing) - 
                                  torch.max(x_i - min_spacing, x_j - min_spacing) - 
                                  torch.tensor(w_i + w_j, device=pos.device))
                
                dist_y = torch.max(torch.tensor(0.0, device=pos.device), 
                                  torch.min(y_i + h_i + min_spacing, y_j + h_j + min_spacing) - 
                                  torch.max(y_i - min_spacing, y_j - min_spacing) - 
                                  torch.tensor(h_i + h_j, device=pos.device))
                
                # If objects are too close but not overlapping
                if dist_x > 0 and dist_y > 0 and (dist_x < min_spacing or dist_y < min_spacing):
                    spacing_penalty = min_spacing - torch.min(dist_x, dist_y)
                    loss = loss + spacing_penalty
                    
        return loss
    
    def optimize(self, iterations=100, lr=0.05):
        """Run SDS optimization to refine object positions"""
        # Don't optimize if there are no objects
        if len(self.object_names) == 0:
            return self.objects_pos, []
            
        # Ensure pos_tensor requires grad and is properly initialized
        self.pos_tensor = torch.tensor(
            [[self.objects_pos[name][0], self.objects_pos[name][1]] 
             for name in self.object_names], 
            dtype=torch.float32,
            requires_grad=True
        )
        
        # Use a smaller learning rate for more stable optimization
        optimizer = optim.Adam([self.pos_tensor], lr=lr)
        
        # Adjust loss weights for better scaling
        self.w_boundary = 50.0  # Increase weight for boundary constraints (was 10.0)
        self.w_overlap = 20.0   # Increase weight for overlap avoidance (was 5.0)
        self.w_wall = 15.0      # Adjusted for wall alignment (was 8.0)
        self.w_spacing = 10.0   # Increased for better spacing (was 3.0)
        
        losses = []
        
        print(f"Optimizing positions for {len(self.object_names)} objects...")
        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            
            # Calculate losses - these now return tensor values
            boundary_loss = self._room_boundary_loss(self.pos_tensor)
            overlap_loss = self._overlap_loss(self.pos_tensor)
            wall_loss = self._wall_alignment_loss(self.pos_tensor)
            spacing_loss = self._spacing_loss(self.pos_tensor)
            
            # Add a small epsilon to each loss to prevent exact zero values
            epsilon = torch.tensor(1e-6, device=self.pos_tensor.device)
            boundary_loss = boundary_loss + epsilon
            overlap_loss = overlap_loss + epsilon
            wall_loss = wall_loss + epsilon
            spacing_loss = spacing_loss + epsilon
            
            # Create differentiable weighted sum using PyTorch operations
            total_loss = boundary_loss * self.w_boundary + \
                         overlap_loss * self.w_overlap + \
                         wall_loss * self.w_wall + \
                         spacing_loss * self.w_spacing
            
            # Ensure loss isn't too small to be meaningful
            # Add a small regularization term based on distance from original positions
            original_pos = torch.tensor(
                [[self.objects_pos[name][0], self.objects_pos[name][1]] 
                for name in self.object_names], 
                dtype=torch.float32,
                device=self.pos_tensor.device
            )
            
            # Add small regularization loss (distance from original)
            reg_loss = torch.sum(torch.norm(self.pos_tensor - original_pos, dim=1)) * 0.01
            total_loss = total_loss + reg_loss
            
            # Store the loss value
            current_loss = total_loss.item()
            losses.append(current_loss)
            
            # Compute gradients and update
            total_loss.backward()
            optimizer.step()
            
            # Early stopping if loss is low enough
            if current_loss < 0.1:
                print(f"Early stopping at iteration {i} with loss {current_loss:.6f}")
                break
                
        # Update object positions with optimized values
        optimized_pos = {}
        for i, obj_name in enumerate(self.object_names):
            new_pos = (int(self.pos_tensor.data[i, 0].item()), 
                      int(self.pos_tensor.data[i, 1].item()))
            optimized_pos[obj_name] = new_pos
            
        return optimized_pos, losses
    
    def visualize_optimization(self, original_pos, optimized_pos, losses=None):
        """Visualize the original and optimized object positions"""
        # Skip visualization if no objects
        if not original_pos or not optimized_pos:
            return
            
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        
        # Plot original positions
        axs[0].imshow(self.room_mask, cmap='gray')
        for obj_name, pos in original_pos.items():
            if obj_name in self.objects_size:
                x, y = pos
                w, h = self.objects_size[obj_name]
                rect = plt.Rectangle((y, x), h, w, fill=False, edgecolor='red', linewidth=2)
                axs[0].add_patch(rect)
                axs[0].text(y + h/2, x + w/2, obj_name, ha='center', va='center', fontsize=8)
        axs[0].set_title('Original Positions')
        
        # Plot optimized positions
        axs[1].imshow(self.room_mask, cmap='gray')
        for obj_name, pos in optimized_pos.items():
            if obj_name in self.objects_size:
                x, y = pos
                w, h = self.objects_size[obj_name]
                rect = plt.Rectangle((y, x), h, w, fill=False, edgecolor='green', linewidth=2)
                axs[1].add_patch(rect)
                axs[1].text(y + h/2, x + w/2, obj_name, ha='center', va='center', fontsize=8)
        axs[1].set_title('Optimized Positions')
        
        # Plot loss curve if available
        if losses and len(losses) > 0:
            # Convert losses to numpy array for easier handling
            losses_array = np.array(losses)
            
            # Check if losses have meaningful values
            if np.any(losses_array > 1e-5):
                # Plot loss curve
                axs[2].plot(range(len(losses)), losses, 'b-', linewidth=2)
                axs[2].set_title('Optimization Loss')
                axs[2].set_xlabel('Iteration')
                axs[2].set_ylabel('Loss')
                axs[2].grid(True, which='both', linestyle='--', alpha=0.6)
                
                # Use log scale if range is appropriate (more than 2 orders of magnitude)
                loss_max = np.max(losses_array)
                loss_min = np.max([np.min(losses_array[losses_array > 0]), 1e-5])  # Avoid zeros
                
                if loss_max / loss_min > 100:
                    axs[2].set_yscale('log')
                
                # Set integer ticks on x-axis
                if len(losses) > 1:
                    axs[2].set_xticks(np.arange(0, len(losses), max(1, len(losses)//5)))
            else:
                # No meaningful loss changes - display a message
                axs[2].text(0.5, 0.5, "No significant loss changes detected\n(values near zero)", 
                           ha='center', va='center', fontsize=12, color='red')
                axs[2].axis('off')
        else:
            axs[2].text(0.5, 0.5, "No loss data available", ha='center', va='center')
            axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("output/optimization/room_positions.png")
        
        # Print final loss information
        if losses and len(losses) > 0:
            if np.any(np.array(losses) > 1e-5):
                print(f"Optimization completed after {len(losses)} iterations. Final loss: {losses[-1]:.6f}")
            else:
                print("Warning: Optimization produced near-zero losses. Check optimization parameters.")
                
        plt.close() 