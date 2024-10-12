# Image_Segmentation


### 🦋 Butterfly Image Segmentation using Masking Techniques
This repository contains an image segmentation project focused on using masking techniques to isolate butterfly images. The dataset was created using Google Images and augmented through Roboflow. The project demonstrates how to use Python libraries, such as OpenCV and pycocotools, to generate segmentation masks from COCO-format annotations.

### Project Overview

This project showcases the process of segmenting butterfly images using masking techniques. A small dataset of 7 images was collected, labeled, and augmented with Roboflow. Python-based tools, such as OpenCV and the COCO API, were used to create binary masks for image segmentation.

## Key Components:
1. Data Collection:
   A small collection of butterfly images was gathered from Google Images.

2. Dataset Preparation with Roboflow:
   The images were annotated in `Roboflow`, generating COCO-format annotations to be used for segmentation tasks.

3. Data Augmentation:
   Using `Roboflow’s` augmentation tools, the dataset was expanded by applying transformations such as rotation, flipping, and scaling.

4. Image Segmentation:
   With the dataset prepared, Python libraries like `OpenCV` and `pycocotools` were used to generate segmentation masks. The masks were created based on the polygonal segmentation data provided in the COCO annotations.


### What is Masking in Image Segmentation?
Masking is a powerful technique used in image segmentation, where specific areas of an image are isolated and distinguished from the rest of the image. This process involves creating a binary mask that overlays the original image, highlighting the regions of interest. Here’s how masking is beneficial in image segmentation: 

## Benefits of Masking:
1. Precision in Object Isolation:
   Masking allows for the precise isolation of objects within an image, which is particularly important in applications like medical imaging, wildlife tracking, and automated inspection.

2. Enhanced Model Training:
   For machine learning models, especially in deep learning, having accurate masks can improve the training process. Models can learn to identify and differentiate between various objects more effectively with clearly defined boundaries.

3. Flexibility Across Applications:
   Masking can be applied in various domains such as autonomous vehicles (to identify pedestrians and obstacles), satellite imagery analysis (to delineate land use), and augmented reality (to separate foreground objects).

4. Facilitates Analysis and Measurements:
  Once objects are masked, it becomes easier to perform quantitative analysis, such as measuring the area of a segmented object, which can be critical in fields like biology and materials science.

5. Improves Visual Clarity:
   Masking enhances the visual representation of objects in images, making it easier to identify specific features that need to be analyzed or monitored.

### Code Explanation
The code provided in this repository walks through the process of creating and visualizing segmentation masks:

1. Loading COCO Annotations:
   The COCO-format annotations are loaded from the provided `_annotations.coco.json` file, which includes both image and segmentation data.
   ```python
   with open('_annotations.coco.json') as f:
        coco_data = json.load(f)
2. Binary Mask Creation Function:
   A helper function create_binary_mask is used to convert segmentation coordinates into a binary mask. The mask is initialized with zeros and filled in using OpenCV’s `fillPoly` function, which creates a mask from the polygon coordinates.
   ```python
   def create_binary_mask(segmentation, image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    for seg in segmentation:
        poly = np.array(seg).reshape((len(seg) // 2, 2))
        cv2.fillPoly(mask, [np.int32(poly)], 1)
    
    return mask
3. Loading and Converting Image:
   The first image from the dataset is loaded using OpenCV, converted to RGB format, and prepared for visualization.
   ```python
   image_info = coco_data['images'][0]  # Get first image info
   image_path = f"{image_info['file_name']}"  # Image path
   image = cv2.imread(image_path)  # Load image
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

4. Extracting Annotations and Generating the Mask:
   The annotations corresponding to the loaded image are extracted from the COCO data, and the binary mask is created. For each annotation, the `create_binary_mask` function is called, and the masks are merged.
   ```python
   annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
   # Create a mask for the image
   mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
   for ann in annotations:
      binary_mask = create_binary_mask(ann['segmentation'], image_info['height'], image_info['width'])
       mask = np.maximum(mask, binary_mask)  # Merge masks
5. Visualizing the Mask and Original Image:
  The binary mask and the original image are displayed using `matplotlib`. This allows users to see the mask overlaid on the butterfly image.

## COCOProcessor Class
The COCOProcessor class is designed to handle the processing of images and their corresponding COCO annotations systematically. Here’s an overview of the class methods and their purposes:
```python
class COCOProcessor:
    def __init__(self, coco_annotation_file, image_dir):
        ...
    
    def _load_coco_annotations(self):
        ...
    
    def create_binary_mask(self, segmentation, image_height, image_width):
        ...
    
    def save_mask(self, mask, image_info):
        ...
    
    def process_image(self, image_id):
        ...
    
    def visualize(self, image, mask_save_path):
        ...
```
### Class Methods
1. `__init__(self, coco_annotation_file, image_dir)`:
   Initializes the processor with the path to the COCO annotation file and the directory where the images are stored. It also loads the COCO data.

2. `_load_coco_annotations(self)`:
   Loads the COCO annotations from the specified JSON file. If the file is not found, it raises an error.

3. `create_binary_mask(self, segmentation, image_height, image_width)`:
   Creates a binary mask from the provided segmentation data. This mask is generated using OpenCV’s polygon filling function.

4. `save_mask(self, mask, image_info`):
   Saves the generated binary mask to a file for further use. The mask is scaled for visibility and saved in the same directory as the original images.

5. `process_image(self, image_id`):
   Processes a specific image based on its ID. It generates and saves the corresponding binary mask by loading the image, extracting annotations, and combining the generated masks.

6. `visualize(self, image, mask_save_path)`:
   Visualizes the original image alongside its generated mask. It loads the saved mask and displays both using `matplotlib`.


## Usage
1. Load Dataset and Annotations:
Ensure you have the COCO annotations and the image files properly linked.

2. Visualize Segmentation:
The provided script visualizes both the original image and its corresponding segmentation mask, allowing you to see how effectively the model segments the butterfly.

3. Apply Further Augmentation/Model Training:
You can expand the dataset further using augmentation techniques or use the binary masks to train a deep learning model for image segmentation.

## Results
Using masking techniques, the project successfully segments butterfly images from their background, with the generated masks providing a clear distinction between the butterfly and the background. Sample results can be viewed within the notebook or using the provided scripts.

### Acknowledgments
* Google Images: For providing the images.
* Roboflow: For simplifying the dataset creation and augmentation process.
* COCO API: For the annotation tools and segmentation techniques.
* Google Colab: For providing free GPU resources to process the data.







