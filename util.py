import numpy as np
import os
import pyvista
import pickle 
import shutil 
import pdb 

""" 
Global Variable Definitions: TODO ~ Customize this to your local paths and do not add data files to source control
"""
SRC_SEG_DIR = "/Users/bencliu/2025_Summer/Iliac/centerline_output_seg"
VESSEL_OPTIONS_HEADERS = ["aorta", "left_la", "right_la", "left_iv", "right_iv"]
VESSEL_OPTIONS_NII = ["aorta.nii.gz", "left_la.nii.gz", "right_la.nii.gz", "left_iv.nii.gz", "right_iv.nii.gz"]
VESSEL_OPTIONS_VTP = ["aorta.vtp", "left_la.vtp", "right_la.vtp", "left_iv.vtp", "right_iv.vtp"]

""" 
WRAPPER FUNCTION: Converts centerline .vtp files to .txt files 
"""
def wrapper_text_conversion():
    subdirs = [d for d in os.listdir(SRC_SEG_DIR) if os.path.isdir(os.path.join(SRC_SEG_DIR, d))]
    for subdir in subdirs:
        for suffix in VESSEL_OPTIONS_HEADERS:
            centerline_dir_ = os.path.join(SRC_SEG_DIR, subdir, 'centerlines', suffix) 
            centerline_path = os.path.join(centerline_dir_, "centerline_full.vtp")
            if not os.path.exists(centerline_path):
                continue 
            vtp_to_txt(centerline_path)

""" 
HELPER FUNCTION: Converts centerline .vtp file to .txt file
"""
def vtp_to_txt(input_path):
    output_path = input_path.replace('.vtp', '.txt')
    mesh = pyvista.read(input_path)
    points = mesh.points
    with open(output_path, 'w') as f:
        for point in points:
            # Write each coordinate with 3 decimal places
            line = f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n"
            f.write(line)

""" 
WRAPPER FUNCTION: Generates midpoint metadata for each vessel mesh file 
"""
def wrapper_depth_file():
    subdirs = [d for d in os.listdir(SRC_SEG_DIR) if os.path.isdir(os.path.join(SRC_SEG_DIR, d))]
    # Process each directory
    for subdir in subdirs:
        depth_dir = os.path.join(SRC_SEG_DIR, subdir, 'metadata')
        if os.path.exists(depth_dir):
            shutil.rmtree(depth_dir)
        os.makedirs(depth_dir, exist_ok=True)
        for suffix in VESSEL_OPTIONS_VTP:
            input_path = os.path.join(SRC_SEG_DIR, subdir, suffix)
            if os.path.exists(input_path):
                depth_dict = midpoint_helper(input_path)
                save_path = os.path.join(depth_dir, f'{suffix.split(".")[0]}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(depth_dict, f)

""" 
HELPER FUNCTION: Generates midpoint metadata for single vessel mesh file 
"""
def midpoint_helper(input_path): 
    mesh = pyvista.read(input_path)
    points = mesh.points
    xyz_coords = points 

    #Find starting points 
    END_REGION_PERCENTAGE = 0.1

    # Find longest axis 
    min_coords_per_axis = np.min(xyz_coords, axis=0)
    max_coords_per_axis = np.max(xyz_coords, axis=0)
    axis_lengths = max_coords_per_axis - min_coords_per_axis
    longest_axis_index = np.argmax(axis_lengths)
    longest_axis_length = axis_lengths[longest_axis_index]
    axis_names = ['X', 'Y', 'Z']
    longest_axis_name = axis_names[longest_axis_index]
    return_dict = {} 

    print(f"\nThe longest axis is: {longest_axis_name} (index {longest_axis_index})")
    print(f"Length of the longest axis: {longest_axis_length:.2f}\n")

    for circular_depth in range(int(min_coords_per_axis[longest_axis_index]), int(max_coords_per_axis[longest_axis_index]), 2):
        points = xyz_coords[abs(xyz_coords[:, longest_axis_index] - circular_depth) < 1]
        midpoint = np.mean(points, axis=0)
        print(f"Depth: {circular_depth}, Midpoint: [{midpoint[0]:.2f}, {midpoint[1]:.2f}, {midpoint[2]:.2f}]")
        return_dict[circular_depth] = midpoint.tolist() 

    # --- 2. Identify Collections of Points at the "Ends" of the Longest Axis ---
    for END_REGION_PERCENTAGE in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        end_region_depth = longest_axis_length * END_REGION_PERCENTAGE
        start_end_upper_bound = min_coords_per_axis[longest_axis_index] + end_region_depth
        points_at_starting_end = xyz_coords[xyz_coords[:, longest_axis_index] <= start_end_upper_bound]
        midpoint = np.mean(points_at_starting_end, axis=0)
    
    return return_dict 


""" 
ARCHIVE FUNCTION: Prototyping midpoint generation 
"""
def read_vtp(path="./output_stream/101_mc.vtp"):
    mesh = pyvista.read(path)
    points = mesh.points
    xyz_coords = points 

    #Find starting points 
    END_REGION_PERCENTAGE = 0.1

    # Find longest axis 
    min_coords_per_axis = np.min(xyz_coords, axis=0)
    max_coords_per_axis = np.max(xyz_coords, axis=0)
    axis_lengths = max_coords_per_axis - min_coords_per_axis
    longest_axis_index = np.argmax(axis_lengths)
    longest_axis_length = axis_lengths[longest_axis_index]
    axis_names = ['X', 'Y', 'Z']
    longest_axis_name = axis_names[longest_axis_index]

    print(f"\nThe longest axis is: {longest_axis_name} (index {longest_axis_index})")
    print(f"Length of the longest axis: {longest_axis_length:.2f}\n")

    for circular_depth in range(int(min_coords_per_axis[longest_axis_index]), int(max_coords_per_axis[longest_axis_index]), 2):
        points = xyz_coords[abs(xyz_coords[:, longest_axis_index] - circular_depth) < 1]
        midpoint = np.mean(points, axis=0)
        print(f"Depth: {circular_depth}, Midpoint: [{midpoint[0]:.2f}, {midpoint[1]:.2f}, {midpoint[2]:.2f}]")

    # --- 2. Identify Collections of Points at the "Ends" of the Longest Axis ---
    for END_REGION_PERCENTAGE in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        end_region_depth = longest_axis_length * END_REGION_PERCENTAGE
        # print(f"Calculated depth for defining 'end' regions: {end_region_depth:.2f}\n")

        # Identify points at the "starting" end (minimum side) of the longest axis
        start_end_upper_bound = min_coords_per_axis[longest_axis_index] + end_region_depth
        points_at_starting_end = xyz_coords[xyz_coords[:, longest_axis_index] <= start_end_upper_bound]
        midpoint = np.mean(points_at_starting_end, axis=0)
        print(f"Midpoint: {midpoint}")

    # Identify points at the "finishing" end (maximum side) of the longest axis
    finish_end_lower_bound = max_coords_per_axis[longest_axis_index] - end_region_depth
    points_at_finishing_end = xyz_coords[xyz_coords[:, longest_axis_index] >= finish_end_lower_bound]

    print(f"Number of points found at the '{longest_axis_name}-min' end: {len(points_at_starting_end)}")
    print(f"Number of points found at the '{longest_axis_name}-max' end: {len(points_at_finishing_end)}\n")

    # --- 3. Calculate the Midpoint (Centroid) for Each Collection of "End" Points ---
    start_midpoint = np.mean(points_at_starting_end, axis=0)
    end_midpoint = np.mean(points_at_finishing_end, axis=0)
    highlight_points = [start_midpoint, end_midpoint]
    return highlight_points 

""" 
ARCHIVE FUNCTION: Prototyping vtp file reading 
"""
def read_cl(path="./output_stream/centerline_90.vtp"):
    mesh = pyvista.read(path)
    return mesh.points 

def wrapper(): 
    wrapper_depth_file()
    pdb.set_trace() 
    wrapper_text_conversion()
    pdb.set_trace() 

if __name__ == "__main__":
    wrapper()