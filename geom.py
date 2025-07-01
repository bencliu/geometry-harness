from vmtk import pypes
from vmtk import vmtkscripts
import pdb 
import vtk
import os 
import pickle 

# source ~/vmtk/pkgs/conda-4.5.0-py36_0/bin/activate 

""" 
Global Variable Definitions: TODO ~ Customize this to your local paths and do not add data files to source control
"""
SRC_SEG_DIR = "/Users/bencliu/2025_Summer/Iliac/centerline_output_seg"
VESSEL_OPTIONS_HEADERS = ["aorta", "left_la", "right_la", "left_iv", "right_iv"]
VESSEL_OPTIONS_NII = ["aorta.nii.gz", "left_la.nii.gz", "right_la.nii.gz", "left_iv.nii.gz", "right_iv.nii.gz"]
VESSEL_OPTIONS_VTP = ["aorta.vtp", "left_la.vtp", "right_la.vtp", "left_iv.vtp", "right_iv.vtp"]


""" 
HELPER FUNCTION: Clean mesh to filter for largest contiguous region and remove floaters 
"""
def clean_vessel(input_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_path)
    reader.Update()

    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(reader.GetOutputPort())
    connectivityFilter.SetExtractionModeToLargestRegion()
    connectivityFilter.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(input_path)
    writer.SetInputData(connectivityFilter.GetOutput())
    writer.Write()

""" 
TESTER FUNCTION (self-contained): <image_reader> and <marching_cubes> test 
"""
def image_reader_test(nii_gz_input_path):
    reader = vmtkscripts.vmtkImageReader()
    reader.InputFileName = nii_gz_input_path 
    reader.format = "dicom" 
    reader.Execute() 

    cubes = vmtkscripts.vmtkMarchingCubes() 
    cubes.Image = reader.Image   
    cubes.Level = 0.5 
    cubes.Execute() 
    pdb.set_trace() 


""" 
WRAPPER FUNCTION: Performs mesh construction, smoothing, and file conversions over pre-specified scans in the global file path 
"""
def refined_base_wrapper():
    subdirs = [d for d in os.listdir(SRC_SEG_DIR) if os.path.isdir(os.path.join(SRC_SEG_DIR, d))]
    
    # Process each directory
    for subdir in subdirs:
        for suffix in VESSEL_OPTIONS_NII:   
            input_path = os.path.join(SRC_SEG_DIR, subdir, suffix)
            if os.path.exists(input_path):
                refined_base(input_path)

""" 
HELPER FUNCTION: Performs mesh construction, smoothing, and file conversions over single segmentation 
"""
def refined_base(input_path_nii_gz):
    # Initalize save variables
    save_dir = os.path.dirname(input_path_nii_gz)
    suffix = input_path_nii_gz.split('/')[-1].split('.')[0]
    output_save_path = os.path.join(save_dir, f'{suffix}.vtp')
    if os.path.exists(output_save_path):
        return 

    # Initalize reader
    reader = vmtkscripts.vmtkImageReader()
    reader.InputFileName = input_path_nii_gz 
    reader.format = "dicom" 
    reader.Execute() 

    # Mesh construction 
    try:
        print("Mesh construction")
        cubes = vmtkscripts.vmtkMarchingCubes() 
        cubes.Image = reader.Image   
        cubes.Level = 0.5 
        cubes.Execute()
        mySurface = cubes.Surface  

        # Smoothing
        print("Smoothing")
        mySmoother = vmtkscripts.vmtkSurfaceSmoothing()
        mySmoother.Surface = mySurface
        mySmoother.PassBand = 0.1
        mySmoother.NumberOfIterations = 30
        mySmoother.Execute()

        # Write to .vtp file 
        print("Writing to .vtp file")
        myWriter = vmtkscripts.vmtkSurfaceWriter()
        myWriter.Surface = mySmoother.Surface
        myWriter.OutputFileName = output_save_path
        myWriter.Execute()
        clean_vessel(myWriter.OutputFileName)
    except:
        print("Error: ", input_path_nii_gz)
        return 


""" 
WRAPPER FUNCTION: Performs centerline construction over smoothed mesh files in the global file path 
"""
def refined_centerline_wrapper(): 
    subdirs = [d for d in os.listdir(SRC_SEG_DIR) if os.path.isdir(os.path.join(SRC_SEG_DIR, d))]

    for i, subdir in enumerate(subdirs):
        #Initialize global save variables 
        centerline_dir = os.path.join(SRC_SEG_DIR, subdir, 'centerlines')
        if not os.path.exists(centerline_dir):
            os.makedirs(centerline_dir)
        for suffix, input_suffix in zip(VESSEL_OPTIONS_HEADERS, VESSEL_OPTIONS_VTP):
            # Initialize save variables 
            input_path = os.path.join(SRC_SEG_DIR, subdir, input_suffix) 
            centerline_dir_ = os.path.join(SRC_SEG_DIR, subdir, 'centerlines', suffix) 
            os.makedirs(centerline_dir_, exist_ok=True)
            
            # Load metadata 
            metadata_path = os.path.join(SRC_SEG_DIR, subdir, 'metadata', f'{suffix}.pkl')
            if not os.path.exists(metadata_path):
                continue 
            metadata_dict = pickle.load(open(metadata_path, 'rb'))
            if not os.path.exists(input_path):
                continue 

            # Call centerline construction helper 
            refined_centerline_base(input_path, centerline_dir_, metadata_dict)
        print("________________________________________________________")
        print(f"Completed {i+1}/{len(subdirs)} subdirs")

""" 
HELPER FUNCTION: Performs centerline construction over single smoothed mesh file 
"""
def refined_centerline_base(input_path, save_folder, metadata_dict):
    try:
        # Initalize reader 
        reader = vmtkscripts.vmtkSurfaceReader() 
        reader.InputFileName = input_path
        reader.ReadVTKXMLSurfaceFile()

        # Read midpoint metadata 
        midpoint_dict = metadata_dict
        depth_range = range(min(midpoint_dict.keys()), max(midpoint_dict.keys()), 4)
        source_points = midpoint_dict[depth_range[0]] 
        target_points = midpoint_dict[depth_range[-1]] 

        # Call centerline construction 
        cl_module = vmtkscripts.vmtkCenterlines()
        cl_module.Surface = reader.Surface 
        cl_module.SeedSelectorName = 'pointlist'
        cl_module.SourcePoints = source_points
        cl_module.TargetPoints = target_points
        cl_module.Execute()

        # Write to .vtp file 
        myWriter = vmtkscripts.vmtkSurfaceWriter()
        myWriter.Surface = cl_module.Centerlines
        myWriter.OutputFileName = f'{save_folder}/centerline_full.vtp'  
        myWriter.Execute() 
        print("Complete")  
    except:
        print("Error: ", input_path)

def tester():
    refined_centerline_wrapper() # Run first
    pdb.set_trace() 
    refined_base_wrapper() # Run second (after midpoint metadata is generated)
    pdb.set_trace()

if __name__ == "__main__":
    tester() 