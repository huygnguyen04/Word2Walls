import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh

# Create output directories first to avoid issues
os.makedirs("output", exist_ok=True)
os.makedirs("output/trajectory", exist_ok=True)
os.makedirs("output/optimization", exist_ok=True)
os.makedirs("output/texture", exist_ok=True)
os.makedirs("output/3d_renders", exist_ok=True)
os.makedirs("output/3d_textures", exist_ok=True)

print("Importing modules...")
# Import core modules
try:
    from floorplan.floorplan_generator import FloorplanGenerator
    from layout.layout_generator import LayoutGenerator
    from trajectory.trajectory_generator import TrajectoryGenerator
    from optimization.sds_optimizer import SDSOptimizer
    from texture.texture_generator import TextureGenerator
    import credentials
except ImportError as e:
    print(f"Error importing core module: {e}")
    print("Please make sure all required dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Add rendering modules with better error handling
render_3d_available = True
try:
    from rendering.mesh_generator import MeshGenerator
    from rendering.texture_mapper import TextureMapper
    from rendering.scene_renderer import SceneRenderer
except ImportError as e:
    print(f"Warning: 3D rendering modules could not be imported: {e}")
    print("3D features will be disabled. Install 3D dependencies if needed.")
    render_3d_available = False
    MeshGenerator = TextureMapper = SceneRenderer = None

def main(prompt="A cozy bedroom with a bed, nightstand and a window", enable_edit=False, enable_3d_render=True, skip_optimization=False):
    """
    Main function to run the Word2Walls pipeline.
    
    Args:
        prompt (str): Text description of the room to generate
        enable_edit (bool): Whether to enable interactive editing
        enable_3d_render (bool): Whether to enable 3D rendering
        skip_optimization (bool): Whether to skip the optimization phase
    """
    print("=== Word2Walls: Text-to-3D Room Layout Generator ===")
    print(f"Prompt: {prompt}")
    
    try:
        print("\n=== Phase 1: Floorplan Generation ===")
        # Create a floorplan generator, the floor plan mesh is stored at ./output, and the fp visualizations are at ./floorplan/output
        floorplanGenerator = FloorplanGenerator(prompt)
        house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers = floorplanGenerator.generate_house_mesh(edit=enable_edit)
    except Exception as e:
        print(f"Error in floorplan generation: {e}")
        traceback.print_exc()
        print("Cannot continue without a floorplan. Exiting.")
        return

    try:
        print("\n=== Phase 2: Room Layout Generation ===")
        # Create a room layout generator
        layoutGenerator = LayoutGenerator(prompt, house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers)
        all_pos, all_siz, all_ang = layoutGenerator.generate_room_objects(edit=enable_edit)
    except Exception as e:
        print(f"Error in layout generation: {e}")
        traceback.print_exc()
        print("Cannot continue without a layout. Exiting.")
        return

    # Store all object data
    room_objects = {}
    object_positions = {}
    object_sizes = {}
    object_angles = {}
    
    for room_idx, (pos, siz, ang) in enumerate(zip(all_pos, all_siz, all_ang)):
        if pos:  # Only if the room has objects
            room_objects[room_idx] = list(pos.keys())
            for obj_name, obj_pos in pos.items():
                full_name = f"room{room_idx}_{obj_name}"
                object_positions[full_name] = obj_pos
                object_sizes[full_name] = siz[obj_name]
                object_angles[full_name] = ang.get(obj_name, 0.0)

    try:
        print("\n=== Phase 3: Trajectory Generation ===")
        # Create a trajectory generator for the house
        trajectoryGenerator = TrajectoryGenerator(border_map_no_doors, object_positions, object_sizes)
        trajectory = trajectoryGenerator.generate_trajectory()
        trajectoryGenerator.visualize_trajectory(save_path="output/trajectory/house_trajectory.png")
        print(f"Generated camera trajectory with {len(trajectory)} viewpoints")
    except Exception as e:
        print(f"Error in trajectory generation: {e}")
        traceback.print_exc()
        print("Continuing without trajectory...")
        trajectory = None

    if not skip_optimization:
        try:
            print("\n=== Phase 4: SDS Optimization ===")
            # Apply SDS optimization to refine object placements
            for room_idx, (pos, siz, ang) in enumerate(zip(all_pos, all_siz, all_ang)):
                if not pos:  # Skip empty rooms
                    continue
                    
                print(f"Optimizing room {room_idx}...")
                room_mask = (border_map_no_doors == room_idx).astype(int)
                
                # Create optimizer
                optimizer = SDSOptimizer(pos, siz, ang, room_mask)
                
                # Increase iterations for better optimization results
                optimized_pos, losses = optimizer.optimize(iterations=80)
                
                # Check if losses are all zero or very close to zero
                if losses and np.any(np.array(losses) > 1e-5):
                    # Visualize optimization with improved display
                    optimizer.visualize_optimization(pos, optimized_pos, losses)
                    print(f"  - Initial layout score: {losses[0]:.4f}")
                    print(f"  - Final layout score: {losses[-1]:.4f}")
                    print(f"  - Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
                    
                    # Update positions
                    all_pos[room_idx] = optimized_pos
                    
                    # Update the object positions dictionary
                    for obj_name, obj_pos in optimized_pos.items():
                        full_name = f"room{room_idx}_{obj_name}"
                        object_positions[full_name] = obj_pos
                else:
                    print(f"Warning: Optimization for room {room_idx} produced zero or near-zero losses.")
                    print("This may indicate an issue with the optimization process.")
                    # Still update positions but flag the potential issue
                    all_pos[room_idx] = optimized_pos
                    
                    # Update the object positions dictionary
                    for obj_name, obj_pos in optimized_pos.items():
                        full_name = f"room{room_idx}_{obj_name}"
                        object_positions[full_name] = obj_pos
                    
                    # Create a simple visualization to show before/after
                    plt.figure(figsize=(15, 5))
                    
                    # Before optimization
                    plt.subplot(1, 3, 1)
                    # Draw room mask
                    plt.imshow(room_mask, cmap='gray', alpha=0.3)
                    # Draw objects
                    for obj_name, obj_pos in pos.items():
                        obj_size = siz[obj_name]
                        rect = plt.Rectangle((obj_pos[0]-obj_size[0]/2, obj_pos[1]-obj_size[1]/2), 
                                           obj_size[0], obj_size[1], 
                                           edgecolor='red', facecolor='none', linewidth=2, alpha=0.7)
                        plt.gca().add_patch(rect)
                        plt.text(obj_pos[0], obj_pos[1], obj_name, 
                                ha='center', va='center', fontsize=8, color='blue')
                    plt.title(f"Room {room_idx} - Before")
                    plt.axis('equal')
                    
                    # After optimization
                    plt.subplot(1, 3, 2)
                    # Draw room mask
                    plt.imshow(room_mask, cmap='gray', alpha=0.3)
                    # Draw objects
                    for obj_name, obj_pos in optimized_pos.items():
                        obj_size = siz[obj_name]
                        rect = plt.Rectangle((obj_pos[0]-obj_size[0]/2, obj_pos[1]-obj_size[1]/2), 
                                           obj_size[0], obj_size[1], 
                                           edgecolor='green', facecolor='none', linewidth=2, alpha=0.7)
                        plt.gca().add_patch(rect)
                        plt.text(obj_pos[0], obj_pos[1], obj_name, 
                                ha='center', va='center', fontsize=8, color='blue')
                    plt.title(f"Room {room_idx} - After")
                    plt.axis('equal')
                    
                    # Empty loss plot with message
                    plt.subplot(1, 3, 3)
                    plt.text(0.5, 0.5, "No significant loss change\nOptimization may require tuning", 
                             ha='center', va='center', fontsize=12, color='red')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"output/optimization/room{room_idx}_optimization_alert.png")
                    plt.close()
        except Exception as e:
            print(f"Error in optimization: {e}")
            traceback.print_exc()
            print("Continuing without optimization...")

    try:
        print("\n=== Phase 5: Texture Generation ===")
        # Create a texture generator with proper parameters
        textureGenerator = TextureGenerator(prompt, border_map_no_doors, room_name_dict, object_positions)
        
        # Generate textures for each room
        room_textures = {}
        for room_idx in room_name_dict.keys():
            if room_idx in room_name_dict:  # Skip unknown rooms
                room_name = room_name_dict[room_idx]
                try:
                    texture = textureGenerator.render_textured_room(room_idx)
                    if texture is not None:
                        room_textures[room_idx] = texture
                except Exception as e:
                    print(f"Error generating texture for room {room_idx}: {e}")
    except Exception as e:
        print(f"Error in texture generation: {e}")
        traceback.print_exc()
        print("Continuing without textures...")
        room_textures = {}

    # 3D rendering - only if available and enabled
    if enable_3d_render and render_3d_available and MeshGenerator is not None:
        try:
            print("\n=== Phase 6: 3D Mesh Generation ===")
            # Create 3D mesh generator
            meshGenerator = MeshGenerator(house_v, house_f, border_map_no_doors, room_name_dict)
            
            # Generate the house mesh by processing each room
            print("Generating combined house mesh...")
            combined_mesh = None
            for room_idx in room_name_dict.keys():
                try:
                    print(f"Processing room {room_idx}...")
                    room_mesh = meshGenerator.generate_room_mesh(room_idx=room_idx)
                    if room_mesh is not None and room_mesh.vertices.shape[0] > 0:
                        if combined_mesh is None:
                            combined_mesh = room_mesh
                        else:
                            combined_mesh = trimesh.util.concatenate([combined_mesh, room_mesh])
                except Exception as e:
                    print(f"Error generating mesh for room {room_idx}: {e}")
            
            # Add furniture meshes
            print("\n=== Phase 7: Adding 3D Furniture ===")
            try:
                # Use add_furniture_to_mesh method which accepts all positions, sizes and angles arrays
                meshGenerator.add_furniture_to_mesh(all_pos, all_siz, all_ang)
                
                # Export mesh after furniture is added
                meshGenerator.export_mesh("output/3d_renders/house_mesh.obj")
                
                # Important: For rendering, we need to use the mesh that's stored in the meshGenerator
                if meshGenerator.mesh is None:
                    print("Warning: MeshGenerator mesh is None. Creating a default mesh for rendering.")
                    # Create a simple box mesh if none exists
                    default_mesh = trimesh.creation.box(extents=[10, 10, 3])
                    meshGenerator.mesh = default_mesh
                
                # The mesh to use for rendering is the one in meshGenerator
                final_mesh = meshGenerator.mesh
            except Exception as e:
                print(f"Error adding furniture: {e}")
                traceback.print_exc()
                # Create a fallback mesh if furniture addition fails
                if combined_mesh is not None:
                    final_mesh = combined_mesh
                else:
                    print("Creating a default mesh for rendering as fallback...")
                    final_mesh = trimesh.creation.box(extents=[10, 10, 3])
            
            # Apply textures if available
            if textureGenerator and hasattr(textureGenerator, 'textures') and textureGenerator.textures:
                try:
                    print("\n=== Phase 8: 3D Texture Mapping ===")
                    textureMapper = TextureMapper(final_mesh, textureGenerator.textures, border_map_no_doors)
                    textured_mesh = textureMapper.apply_textures()
                    
                    # Make sure textured_mesh is not None
                    if textured_mesh is None:
                        print("Warning: TextureMapper produced None mesh. Using untextured mesh instead.")
                        textured_mesh = final_mesh
                    
                    # Make sure TextureMapper has the proper export method
                    if hasattr(textureMapper, 'export_mesh'):
                        textureMapper.export_mesh("output/3d_textures/textured_house.obj")
                    elif hasattr(textureMapper, 'save_mesh'):
                        textureMapper.save_mesh("output/3d_textures/textured_house.obj")
                    else:
                        # Default fallback if method doesn't exist
                        textured_mesh.export("output/3d_textures/textured_house.obj")
                    
                    # Render final scene
                    print("\n=== Phase 9: 3D Scene Rendering ===")
                    # Verify the mesh has valid bounds before rendering
                    if hasattr(textured_mesh, 'bounds') and textured_mesh.bounds is not None and textured_mesh.vertices.shape[0] > 0:
                        renderer = SceneRenderer(textured_mesh)
                        
                        # Use available rendering methods instead of 'render_scene'
                        # First, render a simple view
                        print("Rendering simple scene view...")
                        renderer.render_simple_scene(os.path.join("output/3d_renders", "simple_view.png"))
                        
                        # Then, render multiple views
                        print("Rendering multiple views...")
                        renderer.render_multiple_views(os.path.join("output/3d_renders", "view"))
                        
                        # If we have a trajectory, render a turntable animation
                        if trajectory is not None:
                            print("Creating turntable animation...")
                            renderer.render_turntable_animation(num_frames=30, 
                                                                output_path=os.path.join("output/3d_renders", "turntable.gif"))
                    else:
                        print("Warning: Textured mesh has no valid bounds. Creating a default rendering.")
                        renderer = SceneRenderer()  # Use default scene
                        renderer.render_simple_scene(os.path.join("output/3d_renders", "simple_view.png"))
                except Exception as e:
                    print(f"Error in texture mapping or rendering: {e}")
                    traceback.print_exc()
                    print("Continuing without textured 3D render...")
            else:
                # Simple render without textures
                print("\n=== Phase 8: 3D Scene Rendering (without textures) ===")
                
                # Verify the mesh has valid bounds before rendering
                if final_mesh is not None and hasattr(final_mesh, 'bounds') and final_mesh.bounds is not None and final_mesh.vertices.shape[0] > 0:
                    renderer = SceneRenderer(final_mesh)
                    
                    # Use available rendering methods instead of 'render_scene'
                    # First, render a simple view
                    print("Rendering simple scene view...")
                    renderer.render_simple_scene(os.path.join("output/3d_renders", "simple_view.png"))
                    
                    # Then, render multiple views
                    print("Rendering multiple views...")
                    renderer.render_multiple_views(os.path.join("output/3d_renders", "view"))
                    
                    # Render a turntable animation
                    print("Creating turntable animation...")
                    renderer.render_turntable_animation(num_frames=30, 
                                                        output_path=os.path.join("output/3d_renders", "turntable.gif"))
                else:
                    print("Warning: Mesh has no valid bounds. Creating a default rendering.")
                    renderer = SceneRenderer()  # Use default scene
                    renderer.render_simple_scene(os.path.join("output/3d_renders", "simple_view.png"))
        except Exception as e:
            print(f"Error in 3D rendering: {e}")
            traceback.print_exc()
            print("Continuing without 3D renders...")

    print("\n=== Generation Complete ===")
    print(f"Generated house from prompt: '{prompt}'")
    print("Results are saved in the 'output' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word2Walls: Text-to-3D Room Layout Generator")
    parser.add_argument("--prompt", type=str, default="A cozy bedroom with a bed, nightstand and a window",
                        help="Text description of the room to generate")
    parser.add_argument("--edit", action="store_true", help="Enable interactive editing")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D rendering")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip the optimization phase")
    
    args = parser.parse_args()
    
    main(
        prompt=args.prompt,
        enable_edit=args.edit,
        enable_3d_render=not args.no_3d,
        skip_optimization=args.skip_optimization
    )