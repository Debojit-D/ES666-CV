import pdb
import glob
import importlib.util
import os
import cv2

# Define the absolute path to the Images directory
base_dir = os.getcwd()
image_dir = os.path.join(base_dir, 'Images')
path = os.path.join(image_dir, '*')

# List all subdirectories in the src directory
all_submissions = glob.glob(os.path.join(base_dir, 'src', '*'))
os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)

for idx, algo in enumerate(all_submissions):
    algo_name = os.path.basename(algo)
    print(f"**************** Running Awesome Stitcher developed by: {algo_name} | {idx + 1} of {len(all_submissions)} ********************")
    
    try:
        # Prepare module name and path to the stitcher.py file
        module_name = f"{algo_name}_stitcher"
        filepath = os.path.join(algo, 'stitcher.py')
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Define focal lengths and flags based on image folder names
        focal_lengths = {'I1': None, 'I2': 800, 'I3': 650, 'I4': None, 'I5': None, 'I6': 700}
        Flags = {'I1': 0, 'I2': 1, 'I3': 1, 'I4': 0, 'I5': 0, 'I6': 1}
        
        for impaths in glob.glob(path):
            PanaromaStitcher = getattr(module, 'PanaromaStitcher')
            
            # Collect images from the specified path
            image_files = glob.glob(os.path.join(impaths, '*'))
            img_key = os.path.basename(impaths)
            
            # Initialize PanaromaStitcher with the correct focal length and flag
            inst = PanaromaStitcher(
                image_files=image_files,
                focal_length=focal_lengths.get(img_key, None),
                Flag=Flags.get(img_key, 0)
            )
            print(f"\t\t Processing... {impaths}")
            
            # Perform stitching
            stitched_image, homography_matrix_list = inst.stitch_images()
            
            # Save the stitched image in the results directory
            result_folder = os.path.join(base_dir, 'results', img_key)
            os.makedirs(result_folder, exist_ok=True)
            outfile = os.path.join(result_folder, f'{module_name}.png')
            
            cv2.imwrite(outfile, stitched_image)
            print(homography_matrix_list)
            print(f'Panorama saved at: {outfile}\n\n')
            
    except Exception as e:
        print(f"Oh No! My implementation encountered this issue\n\t{e}\n\n")
