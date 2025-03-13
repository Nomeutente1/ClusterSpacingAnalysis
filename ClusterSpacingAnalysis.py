import cv2
import csv
import heapq
import os
import scipy

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


from IPython.display import clear_output
from moviepy import VideoFileClip
from PIL import Image, ImageTk, ImageFilter
from skimage.morphology import skeletonize
from scipy.ndimage import generate_binary_structure, label, distance_transform_edt, uniform_filter1d
from scipy.spatial import distance_matrix
from tkinter import filedialog, messagebox, ttk, simpledialog

def stack_creation():
    """ stack_creation() 
    The function creates an image stack starting from a movie and saves all the frames in a subfolder in the same folder of the movie.

    Parameters:
    -----------
    @None

    Output:
    -----------
    @Subfolder_stack: Folder that contains all the frames
    """
    def select_file():
        # Create a root window (it will not be displayed)
        root = tk.Tk()
    
        # Open the file dialog and allow the user to select a file
        file_path = filedialog.askopenfilename(title="Select a file")
        
        folder_path = os.path.dirname(file_path) # Save the folder in which the file is located
    
        root.destroy()
        return file_path, folder_path # Return both the file path and the folder path

    global subfolder_stack, folder_path, file_path
    
    file_path, folder_path = select_file() # Calling the function to extract the variables

    folder_stack = "Image Stack" # Creating a subfolder for the image stack
    subfolder_stack = os.path.join(folder_path, folder_stack).replace("\\", "/")
    # Checking if the folder exist 
    if not os.path.exists(subfolder_stack):
        os.makedirs(subfolder_stack)
    else:
        pass

    clip = VideoFileClip(file_path) # Separating the movie in frames and saving them in subfolder_stack
    time_length = clip.duration
    fps = clip.fps
    clip.subclipped(0, time_length).write_images_sequence(subfolder_stack + '//' + 'frame%04d.png', fps=fps)
    clip.close()

def frame_selection(folder_stack):
    """ frame_selection(folder_stack):
    Select only 4 frames from the entire stack. The selection starts from the last one in order to avoid the first quarter of the evolution, in which 
    the evolution is absent. 

    Parameters:
    -----------
    @folder_stack: folder that contains all frames

    Output:
    -----------
    @selected_frames: Frames obtained from the selection
    """
    # List all the images and get their total number
    all_images = os.listdir(folder_stack)
    
    total_images = len(all_images)
    
    # Calculate the equispaced image indices
    step = total_images // 4
    indices = [step * i for i in range(1, 5)]  
    
    # Ensure the last image is included
    indices[-1] = total_images - 1
    
    # Get the selected images
    selected_frames = [all_images[i] for i in indices]
    
    return selected_frames

def click_coordinates(image):
    """
    Extract the coordinates of two points from an image via mouse clicks.
    
    Parameters:
    - image: A copy of the original image in RGB format.
    
    Returns:
    - A tuple containing the coordinates of the two selected points.
    """ 
    # Copy of the image for manipulation
    image_copy = image.copy()
    
    # List to store the coordinates of the clicked points
    clicked_points = []
    
    # Define the mouse click event handler
    def click_event(event, x, y, flags, params):
        nonlocal clicked_points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the coordinates of the left mouse button down event
            clicked_points.append((x, y))
            
            # Draw a small circle at the clicked point
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            
            # Update the displayed image
            cv2.imshow("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", image_copy)
            
            # Check if two points have been selected
            if len(clicked_points) == 2:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return clicked_points # Return the coordinates upon selection completion
    
    
    # Set up the mouse callback
    cv2.imshow("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", image_copy)
    cv2.setMouseCallback("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", click_event)
    
    # Wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clicked_points
def calculate_rescaling_factor(filepath):
    """
    Calculate the rescaling factor of an image based on the distance between two points.
    
    Parameters:
    - filepath: Filepath of the image to analyze.
    
    Returns:
    - rescaling_factor: The rescaling factor calculated from the real distance and the distance derived from the coordinates.
    """
    # Load the image
    image = cv2.imread(filepath)

    # Extract coordinates of two points
    coordinates = click_coordinates(image)
    line_length = np.sqrt((coordinates[1][0] - coordinates[0][0])**2)
    
    # Prompt user for scalebar length and order of magnitude
    scalebar = float(input("Insert scalebar length in real units: "))
    clear_output(wait=True)

    rescaling_factor = scalebar / line_length

    return rescaling_factor

def binarization(image, thr):
    """ binarization(image, threshold) -> binary image, threshold values
    GUI for easily binarize an 8-bit image (signal turns to 1, background turns to 0. If threshold is [0,0], the GUI displays the image and permitts
    to select the thersholding values. If threshold is not [0,0] due to previous iterations, function give directly the binarized image.

    Parameters:
    -----------
    @image: 8-bit image in grayscale
    @threshold: thresholding values [lower, upper]. 
        If they are both 0, GUI displays the image to select new values. If they are not 0, function gives directly the image

    Output:
    -----------
    @bw_image: binarized image.
        The signal is 1, background is 0
    @thr: thresholding values obtained from the GUI
    """
    # If thresholding values are both zeroes, GUI display the image
    # E.g, first iteration in a for-loop
    if thr == [0,0]:
        def range_threshold(image, thr):
            # Function that binarize the image. Signal between thresholding values is 255, background is 0
            output_img = np.zeros_like(image)
        
            # Iterating over rows and columns
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if thr[0] <= image[y, x] <= thr[1]:
                        output_img[y, x] = 255
                    else:
                        output_img[y, x] = 0
            # Binary image (signal = 255, background = 0)
            return output_img
        
        def create_mask(bw_image, image):
            # Mask will be overlayed with the image to easily display the binarization process
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            transparency = 0.6
            red_colour = (255, 0, 0)
            coloured_mask = np.zeros((bw_image.shape[0], bw_image.shape[1], 3), dtype=np.uint8)
            coloured_mask[bw_image == 255] = red_colour
            overlay = cv2.addWeighted(image, transparency, coloured_mask, 1 - transparency, 0)
            return overlay
        
        def update_image():
            # Get the threshold values from the sliders
            lower_thresh = lower_slider.get()
            upper_thresh = upper_slider.get()
        
            # Apply the range thresholding function
            bw_image = range_threshold(image, [lower_thresh, upper_thresh])
        
            # Create a mask with the thresholded image
            overlay = create_mask(bw_image, image)
        
            # Convert the result to a format suitable for Tkinter
            overlay_pil = Image.fromarray(overlay)
            overlay_tk = ImageTk.PhotoImage(overlay_pil)
        
            # Update the label with the new image
            label.config(image=overlay_tk)
            label.image = overlay_tk
        
        def on_slider_change():
            # Takes the thresholding values from the sliders
            lower_thresh = lower_slider.get()
            upper_thresh = upper_slider.get()
        
            # Ensure the upper threshold doesn't go below the lower threshold
            if lower_thresh > upper_thresh:
                upper_slider.set(lower_thresh)
        
            # Ensure the lower threshold doesn't go above the upper threshold
            if upper_thresh < lower_thresh:
                lower_slider.set(upper_thresh)
        
            # Update the image based on the current slider values
            update_image()
        
        def on_done():
            global bw_image, lower_thresh, upper_thresh
            # Get the final threshold values and binarized image
            lower_thresh = lower_slider.get()
            upper_thresh = upper_slider.get()
            bw_image = range_threshold(image, [lower_thresh, upper_thresh])
            #bw_image[bw_image == 255] = 1
            # Close the window
            root.destroy()
        
            # Return the threshold values and binarized image
            return lower_thresh, upper_thresh, bw_image
        
        # Create the main window
        root = tk.Tk()
        root.title("Thresholding Optimization")
        root.geometry("1300x1400")
        
        # Create a frame for the left part (image)
        left_frame = tk.Frame(root, width=1024, height=1024)
        left_frame.grid(row=0, column=0, padx=10, pady=10)
        
        # Create a frame for the right part (sliders and button)
        right_frame = tk.Frame(root, width=600, height=800)
        right_frame.grid(row=0, column=1, padx=10, pady=10)
        
        # Add a canvas in the left frame to display the image
        label = tk.Label(left_frame)
        label.pack()
        
        # Add sliders for lower and upper threshold values in the right frame
        lower_slider = tk.Scale(right_frame, from_=0, to=255, orient="horizontal", label="Lower", command=lambda val: on_slider_change())
        lower_slider.set(0)  # initial value
        lower_slider.pack(pady=10)
        
        upper_slider = tk.Scale(right_frame, from_=0, to=255, orient="horizontal", label="Upper", command=lambda val: on_slider_change())
        upper_slider.set(255)  # initial value
        upper_slider.pack(pady=10)
        
        # Add a "Done" button in the right frame
        button_done = tk.Button(right_frame, text="Done", command=on_done)
        button_done.pack(pady=5)
        
        # Start the initial image update
        update_image()
        
        # Run the GUI
        root.mainloop()

        # Saves the thresholding values in thr
        thr[0] = lower_thresh
        thr[1] = upper_thresh
        return bw_image, thr

    else:
        # If thresholding values are not both 0, use them to binarize the image
        def range_threshold(image, thr):
            output_img = np.zeros_like(image)
        
            # Iterating over rows and columns
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if thr[0] <= image[y, x] <= thr[1]:
                        output_img[y, x] = 255
                    else:
                        output_img[y, x] = 0
        
            return output_img
            
        bw_img = range_threshold(image, thr)
        #bw_image[bw_image == 255] = 1
        return bw_img

def ero_dil_processing(data, n_erode, n_dilate, operation_history):
    """ ero_dil_processing(data, number of erosion, number of dilation, operation's history)
    The function performs erosion/dilation processes to remove spurious pixels. Operation history takes notes of all the operation performed
    in order to reuse the same order in successive iterations.

    Parameters:
    -----------
    @data: 8-bit image array
    @n_erode: 0 for the first iterations.
        For successive iterations, count how many times the erosion process was performed
    @n_dilatee: 0 for the first iterations.
        For successive iterations, count how many times the dilation process was performed
    @operation_history: Takes notes of all the operation performed.
        For successive iterations, perform the operations in the same order

    Output:
    -----------
    First iteration: 
    @corrected_image: 8-bit binarized image (signal: 1, bg: 0) after erosio/dilation
    @n_erode: Number of times erosion process was performed
    @n_dilate: Number of times dilation process was performed
    @operation_history: Takes notes of all the operation performed.

    Successive iterations: 
    @corrected_image: 8-bit binarized image (signal: 1, bg: 0) after erosio/dilation
    """
    
    if n_erode == 0 and n_dilate == 0 and operation_history == []: # First iteration
        # Function to update image on the canvas
        def update_image_on_canvas(img):
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.image = photo  # Keeping reference
        
        # Function to perform image erosion
        def perform_erosion():
            nonlocal image, n_erode
            # Erosion can be simulated by using a small kernel in a filter
            image = image.filter(ImageFilter.MinFilter(size=3))  # Size 3 for erosion
            image_history.append(image)  # Saving modified image
            operation_history.append('erode')  # Saving history
            n_erode += 1 # Updating counter
            update_image_on_canvas(image) # Updating image
        
        # Function to perform image dilation
        def perform_dilation():
            nonlocal image, n_dilate
            # Dilation can be simulated using a large kernel in a filter
            image = image.filter(ImageFilter.MaxFilter(size=3))  # Size 3 for erosion
            image_history.append(image)  # Saving the modified image
            operation_history.append('dilate') # Saving history
            n_dilate += 1 # Updating counter
            update_image_on_canvas(image) # Updating image
        
        # Function to undo the last operation (return to previous image)
        def undo_last_operation():
            nonlocal image
            if len(image_history) > 1:
                image_history.pop()  # Remove the last image in history
                operation_history.pop() # Remove the last operation in history
                image = image_history[-1]  # Get the previous image
                update_image_on_canvas(image) # Updating image
    
        def on_done():
                nonlocal corrected_image
                # Get the final image
                corrected_image = np.array(image)
                corrected_image[corrected_image == 255] = 1 # Binarization
                # Close the window
                root.destroy()
            
                # Return the image
                return corrected_image, n_erode, n_dilate, operation_history
        
        # Create the main window
        
        root = tk.Tk()
        root.title("Erosion/Dilation adjuster")
        canvas = tk.Canvas(root, width=1100, height=1100)
        canvas.pack(side="left", padx=0, pady=0)
    
        # Labeling window
        image_label = tk.Label(root)
        image_label.pack(side="right", padx=10, pady=10)
        
        # Display the image on canvas
        image = Image.fromarray(data)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo  # Keep a reference to avoid garbage collection
        
        # Create a list to store the image history (for undo functionality)
        image_history = [image]
        
        # Create buttons for Erosion, Dilation, and Undo
        erosion_button = tk.Button(root, text="Erosion", command=perform_erosion)
        erosion_button.pack(side="top", padx=10, pady=10)
        
        dilation_button = tk.Button(root, text="Dilation", command=perform_dilation)
        dilation_button.pack(side="top", padx=10, pady=10)
        
        return_button = tk.Button(root, text="Return", command=undo_last_operation)
        return_button.pack(side="top", padx=10, pady=10)
    
        # Add a "Done" button in the right frame
        button_done = tk.Button(root, text="Done", command=on_done)
        button_done.pack(side="top", padx=10, pady=10)
    
        # Display the initial image on canvas
        update_image_on_canvas(image)
        
        # Start the Tkinter main loop
        root.mainloop()
    
        return corrected_image, n_erode, n_dilate, operation_history

    else: # If n_erode or n_dilate is not zero, the GUI is removed (successive iterations)
        image = Image.fromarray(data)  # Convert numpy array to PIL Image

        for operation in operation_history: # Performing the operations in the same order
            if operation == 'erode':
                image = image.filter(ImageFilter.MinFilter(size=3))  # Erosion
            elif operation == 'dilate':
                image = image.filter(ImageFilter.MaxFilter(size=3))  # Dilation

        # Convert the final image back to a numpy array for the result
        corrected_image = np.array(image)
        corrected_image[corrected_image == 255] = 1

        return corrected_image

def image_processing(file_path, flag):
    """ image_processing(file path)
    The function iterate the process from a grayscale image to a binarized one. First it allows to select a threshold for the binarization, then 
    the function performs erosion/dilation processes
    
    Parameters:
    -----------
    @file_path: Path of the image
    
    Output:
    -----------
    @ed_image: Binarized image after erosion/dilation
    @data: Image in grayscale
    """

    global thr, n_erode, n_dilate, op_hist
    
    if flag: # First iteration
        
        data = np.array(Image.open(file_path)) # Opening the image as array
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        binary_image, thr = binarization(data, [0,0]) # Binarization
        ed_image, n_erode, n_dilate, op_hist = ero_dil_processing(binary_image, 0, 0, []) # Performing erosion/dilation
        flag = False
        
    else:

        data = np.array(Image.open(file_path)) # Opening the image as array
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        binary_image = binarization(data, thr) # Binarization
        ed_image = ero_dil_processing(binary_image, n_erode, n_dilate, op_hist) # Performing erosion/dilation using the same conditions

    return ed_image, data

def ridge_map(bw_image, rescaling, result_path, filename):
    """ ridge_map(binarized image, rescaling factor, folder, file name)
    The function calculates the Euclidean distance map and the ridge map, which is obtained using both the ED map and the skeleton. 

    Parameters:
    -----------
    @bw_image: Binarized image, signal is 1 while background is 0
    @rescaling: Rescaling factor for rescaling the distances
    @result_path: Path of the folder in which the images will be saved
    @filename: Name of the original image (e.g. Frame 01.png)

    Output:
    -----------
    @spacing: Array of distances between each signal point to the nearest background point
    """
    # Calculating euclidean distance map
    distances = scipy.ndimage.distance_transform_edt(bw_image, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None)*rescaling
    skeleton = skeletonize(bw_image).astype(int)
    # Create a new array for the result, initially filled with zeros
    ridge = np.zeros_like(distances)
    
    # Retain the distance values where skeleton is non-zero
    ridge[skeleton > 0] = distances[skeleton > 0]

    plt.figure(1, figsize=(10, 10))  # Adjust figsize
    plt.imshow(distances, cmap='turbo')
    plt.axis('off')  # Completely turn off axes
    plt.colorbar()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Remove white margins
    plt.savefig(f"{result_path}/{filename}_ED_map.png")
    plt.close()  # Close the plot to avoid display

    # Second Figure - result
    plt.figure(2, figsize=(10, 10))
    plt.imshow(ridge, cmap='turbo')
    plt.axis('off')
    plt.colorbar()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.savefig(f"{result_path}/{filename}_Ridge_map.png")
    plt.close()  # Close the plot to avoid display

    return ridge

def distance_calculation(bw_image, rescaling, result_path, filename):
    """ distance_calculation(binarized image, rescaling factor, folder, file name)
    The function calculates the distribution of distances between clusters in an image. 

    Parameters:
    -----------
    @bw_image: Binarized image
    @rescaling: Rescaling factor for rescaling the distances
    @results_path: Path of the folder in which the images will be saved
    @filename: Name of the image (e.g. Frame 01.png)

    Output:
    -----------
    @spacing: Array of the distances in the ridge's points
    """
    r = ridge_map(bw_image, rescaling, result_path, filename) # Calculates the ED map and the ridge map. The output is the ridge map

    spacing = 2*r # Calculates the spacing of the voids between clusters by multiply by a factor 2 the nearest distance obtained

    spacing = spacing[spacing!=0] # Keeping only the non-zero values

    counts, bins = np.histogram(spacing, bins=20) # Calculating histogram 
    smooth_counts = np.convolve(counts, np.ones(5)/5, mode='same') 
    
    plt.figure(figsize=(10, 8)) 
    plt.fill_between(bins[:-1], smooth_counts, step='mid', color='blue', alpha=0.5)  
    plt.step(bins[:-1], smooth_counts, where = 'mid' , color='black', linewidth=2.5)  
    
    plt.title("Distance distribution", fontsize = 25)
    plt.xlabel("Distance (nm)", fontsize = 25)
    plt.ylabel("Counts", fontsize = 25)
    plt.xticks(fontsize = 25)
    # Remove y-axis (including scale)
    plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
    plt.yticks([])  # Remove the y-ticks
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(f"{result_path}/{filename}_Distribution.png")
    plt.close()  # Close the plot to avoid display

    return spacing

def total_distribution_plot(data_set, result_path, filename):
    """ total_distribution_plot(result folder, data, file name)
    This function creates and save the distribution of all the selected frames.

    Parameters:
    -----------
    @data_set: Distances of all the frames
    @result_path: Folder in which the image will be saved
    @filename: Name of the file

    Output:
    -----------
    None
    """

    # Create a single axis for all histograms
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, data in enumerate(data_set):
        # Create the histogram
        counts, bins = np.histogram(data, bins=20)
    
        # Apply Gaussian smoothing to the histogram
        smooth_hist = np.convolve(counts, np.ones(5)/5, mode='same') 
    
        # Plot the step line
        plt.step(bins[:-1], smooth_hist, where='mid', color=colors[i], linewidth=5)
    
        # Set the plot title and labels
        ax.set_title('Total distance distributions', fontsize = 20)
        ax.set_xlabel('Distance (nm)', fontsize = 20)
        ax.set_ylabel('Counts', fontsize = 20)
        ax.set_xlim([0, np.max(data_set[3])])
        # Remove y-axis (including scale)
        plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
        plt.yticks([])  # Remove the y-ticks
        plt.xticks(fontsize = 20)
       
    # Add a legend to the plot
    ax.legend([f'{filename[0]}', f'{filename[1]}', 
                f'{filename[2]}', f'{filename[3]}'],
               loc='upper right', fontsize = 20)         
    
    # Adjust layout and display the plot
    fig.tight_layout()
    plt.savefig(f"{result_path}/ED total distributions.png")
    plt.close()  # Close the plot to avoid display

def nearest_neighbour_approach(bw_image, gs_image, mean, result_path, rescaling, filename):
    """ nearest_neighbour_approach(binarized image, grayscale image, mean value, folder)
    The function calculates the distances' distribution of clusters using a nearest-neighbour approach. First, the function identify all the clusters
    and calculates the contourn of each cluster. Then, calculates the minimum distance between a cluster and 1 or 3 nearest neighbour. Later, the 
    function plots and save the distribution with a grayscale image with blue lines that rappresent the distances.

    Parameters: 
    -----------
    @bw_image: Binarized image obtained from the binarization process. NOTE: the image has the background as signal, thus it must be inverted
    @gs_image: Original image
    @mean: Mean value of the distribution of distances obtained from the Euclidean distance approach. It acts as first point to calculates a cutoff 
    @result_path: Folder in which all the images will be saved

    Output:
    -----------
    @all_distances: Array contaninings all the distances between clusters. 1 nearest neighbour
    @all_distances_3: Array containings all the distances between clusters. 3 nearest neighbour
    """

    img = cv2.bitwise_not(bw_image) # Inverting the image from the ED approach 
    img[img == 255] = 1
    img[img == 254] = 0

    max_distances_per_particle = [1, 3]
    cutoff_distance = np.int16(2*mean) # Selecting a cutoff distance

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Finding contours

    # Initialize a copy of the grayscale image to draw lines on
    overlay_image = np.copy(cv2.cvtColor(gs_image, cv2.COLOR_GRAY2RGB))

    for max_dist in max_distances_per_particle:
        if max_dist == 1:
            # List to hold the minimum distances
            all_distances = []
            
            # Loop over all particle contours
            for i in range(len(contours)):
                min_distances_for_particle = []  # List to store the minimum distances for particle i
                
                for j in range(i + 1, len(contours)):
                    # Get all points on the contours of the two particles
                    points1 = contours[i].reshape(-1, 2)
                    points2 = contours[j].reshape(-1, 2)
                    
                    # Compute distances between all points on contour1 and all points on contour2
                    distances = distance_matrix(points1, points2)
                    
                    # Get the minimum distance between the two contours
                    min_distance = float(np.min(distances) * rescaling)
                    
                    if min_distance <= cutoff_distance:
                        # Only keep up to max_distances_per_particle smallest distances
                        if len(min_distances_for_particle) < max_dist:
                            min_index = np.unravel_index(np.argmin(distances), distances.shape)
                            point1 = tuple(points1[min_index[0]].tolist())  # Convert numpy array to tuple
                            point2 = tuple(points2[min_index[1]].tolist())  # Convert numpy array to tuple
                            # Store the distance along with the corresponding points
                            min_distances_for_particle.append((min_distance, point1, point2))
                            heapq.heapify(min_distances_for_particle)  # Ensure the list is a heap after insertion
                        else:
                            if min_distance < min_distances_for_particle[0][0]:  # Compare with the largest distance in the heap
                                min_index = np.unravel_index(np.argmin(distances), distances.shape)
                                point1 = tuple(points1[min_index[0]].tolist())
                                point2 = tuple(points2[min_index[1]].tolist())
                                heapq.heapreplace(min_distances_for_particle, (min_distance, point1, point2))
                                        
                # After collecting the closest distances, draw the lines based on those distances
                for dist, point1, point2 in min_distances_for_particle:
                    # Draw a line between the closest points on the contours
                    cv2.line(overlay_image, tuple(point1), tuple(point2), (255, 0, 0), 2)  # Blue line
        
                # Add the smallest distances for particle i to the overall distances list
                all_distances.extend([d[0] for d in min_distances_for_particle])  # Extract only the distances for the histogram
        
            plt.figure(figsize=(12, 10))
            plt.hist(all_distances, bins=15, edgecolor='black', color='green')
            # Set the title and label with increased font size
            plt.title("Distance distribution", fontsize=25) # Increase font size for title
            plt.xlabel('Distance (nm)', fontsize=25) # Increase font size for x-axis label
            plt.xticks(fontsize = 25)
            plt.ylabel('Counts', fontsize = 25)
            
            hist_counts, bin_edges = np.histogram(all_distances, bins=15)
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centri dei bin
            
            # Calcola un'interpolazione lineare "a grandi linee"
            plt.plot(bin_midpoints, np.interp(bin_midpoints, bin_midpoints, hist_counts), 'r-', lw=14, label='Trend Line')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
            # Remove y-axis (including scale)
            plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
            plt.yticks([])  # Remove the y-ticks
            plt.savefig(f"{result_path}/{filename}_1nn_distributions.png")
            plt.close()  # Close the plot to avoid display
            cv2.imwrite(f"{result_path}/{filename}_1nn_image.tif", overlay_image)

        else:
            # List to hold the minimum distances
            all_distances_3 = []
            
            # Loop over all particle contours
            for i in range(len(contours)):
                min_distances_for_particle = []  # List to store the minimum distances for particle i
                
                for j in range(i + 1, len(contours)):
                    # Get all points on the contours of the two particles
                    points1 = contours[i].reshape(-1, 2)
                    points2 = contours[j].reshape(-1, 2)
                    
                    # Compute distances between all points on contour1 and all points on contour2
                    distances = distance_matrix(points1, points2)
                    
                    # Get the minimum distance between the two contours
                    min_distance = float(np.min(distances) * rescaling)
                    
                    if min_distance <= cutoff_distance:
                        # Only keep up to max_distances_per_particle smallest distances
                        if len(min_distances_for_particle) < max_dist:
                            min_index = np.unravel_index(np.argmin(distances), distances.shape)
                            point1 = tuple(points1[min_index[0]].tolist())  # Convert numpy array to tuple
                            point2 = tuple(points2[min_index[1]].tolist())  # Convert numpy array to tuple
                            # Store the distance along with the corresponding points
                            min_distances_for_particle.append((min_distance, point1, point2))
                            heapq.heapify(min_distances_for_particle)  # Ensure the list is a heap after insertion
                        else:
                            if min_distance < min_distances_for_particle[0][0]:  # Compare with the largest distance in the heap
                                min_index = np.unravel_index(np.argmin(distances), distances.shape)
                                point1 = tuple(points1[min_index[0]].tolist())
                                point2 = tuple(points2[min_index[1]].tolist())
                                heapq.heapreplace(min_distances_for_particle, (min_distance, point1, point2))
                                        
                # After collecting the closest distances, draw the lines based on those distances
                for dist, point1, point2 in min_distances_for_particle:
                    # Draw a line between the closest points on the contours
                    cv2.line(overlay_image, tuple(point1), tuple(point2), (255, 0, 0), 2)  # Blue line
        
                # Add the smallest distances for particle i to the overall distances list
                all_distances_3.extend([d[0] for d in min_distances_for_particle])  # Extract only the distances for the histogram
        
            plt.figure(figsize=(12, 10))
            plt.hist(all_distances_3, bins=15, edgecolor='black', color='green')
            # Set the title and label with increased font size
            plt.title("Distance distribution", fontsize=25) # Increase font size for title
            plt.xlabel('Distance (nm)', fontsize=25) # Increase font size for x-axis label
            plt.xticks(fontsize = 25)
            plt.ylabel('Counts', fontsize = 25)
            
            hist_counts, bin_edges = np.histogram(all_distances_3, bins=15)
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centri dei bin
            
            # Calcola un'interpolazione lineare "a grandi linee"
            plt.plot(bin_midpoints, np.interp(bin_midpoints, bin_midpoints, hist_counts), 'r-', lw=14, label='Trend Line')
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
            # Remove y-axis (including scale)
            plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
            plt.yticks([])  # Remove the y-ticks
            plt.savefig(f"{result_path}/{filename}_3nn_distributions.png")
            plt.close()  # Close the plot to avoid display
            cv2.imwrite(f"{result_path}/{filename}_3nn_image.tif", overlay_image)

    return all_distances, all_distances_3

def plotting_nearest_neighbour(nn1_data, nn3_data, filename, result_path):
    """ plotting_nearest_neighbour(data 1 neighbour, data 3 neighbours, file name, folder)
    The function plots all the histograms together to improve the data analysis. First it plots the data with 1 nearest neighbour, then plots the data
    for 3 nearest neighbours. Finally, it saves both images in a folder.

    Parameters: 
    -----------
    @nn1_data: Data containings all the distances calculates for all the selected frames. 1 nearest neighbour
    @nn3_data: Data containings all the distances calculates fo all the selected frames. 3 nearest neighbours
    @filename: Name of the file (e.g. Frame 01.png)
    @resulth_path: Folder in which all the images will be saved
    """

    # Create a single axis for all histograms with 1 nearest neighbour
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, data in enumerate(nn1_data):
        # Create the histogram
        counts, bins = np.histogram(data, bins=20)
    
        # Apply Gaussian smoothing to the histogram
        smooth_hist = np.convolve(counts, np.ones(5)/5, mode='same')
        
        # Plot the step line 
        plt.step(bins[:-1], smooth_hist, where='mid', color=colors[i], linewidth=5)
    
        # Set the plot title and labels
        ax.set_title('Nearest neighbour - Total distance distributions', fontsize = 20)
        ax.set_xlabel('Distance (nm)', fontsize = 20)
        ax.set_ylabel('Counts', fontsize = 20)
        ax.set_xlim([0, np.max(data)])
        # Remove y-axis (including scale)
        plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
        plt.yticks([])  # Remove the y-ticks
        plt.xticks(fontsize = 20)
       
    # Add a legend to the plot
    ax.legend([f'{filename[0]}', f'{filename[1]}', 
                f'{filename[2]}', f'{filename[3]}'],
               loc='upper right', fontsize = 20)         
    
    # Adjust layout and display the plot
    fig.tight_layout()
    plt.savefig(f"{result_path}/nn1 total distributions.png")
    plt.close()  # Close the plot to avoid display

    # Create a single axis for all histograms with 3 nearest neighbour
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, data in enumerate(nn3_data):
        # Create the histogram
        counts, bins = np.histogram(data, bins=20)
    
        # Apply Gaussian smoothing to the histogram
        smooth_hist = np.convolve(counts, np.ones(5)/5, mode='same')
        
        # Plot the step line 
        plt.step(bins[:-1], smooth_hist, where='mid', color=colors[i], linewidth=5)
    
        # Set the plot title and labels
        ax.set_title('Nearest neighbour - Total distance distributions', fontsize = 20)
        ax.set_xlabel('Distance (nm)', fontsize = 20)
        ax.set_ylabel('Counts', fontsize = 20)
        ax.set_xlim([0, np.max(data)])
        # Remove y-axis (including scale)
        plt.gca().get_yaxis().set_visible(True)  # Hide the y-axis
        plt.yticks([])  # Remove the y-ticks
        plt.xticks(fontsize = 20)
       
    # Add a legend to the plot
    ax.legend([f'{filename[0]}', f'{filename[1]}', 
                f'{filename[2]}', f'{filename[3]}'],
               loc='upper right', fontsize = 20)         
    
    # Adjust layout and display the plot
    fig.tight_layout()
    plt.savefig(f"{result_path}/nn3 total distributions.png")
    plt.close()  # Close the plot to avoid display

def ClusterSpacingAnalysis_v1(): 
    """ClusterSpacingAnalysis_v1()
    The function takes as input a movie, separate it in a stack of frames and select 4 equispaced frames. Then calculates the distribution of distances
    between clusters with two different approaches: Euclidean distance and nearest neighbour. 

    In the Euclidean distance approach first it's calculated the Euclidean distance map, then the ridge map. The latter is an overlay between the 
    Euclidean map and the skeleton, and provides information regarding the distance between the mean points of the spacing between clusters. 
    Multiplicating this distance by a factor 2 provides the width of the voids between clusters. The function calculates the distribution of these 
    distances (both for the single frame and all together) and some metrices relatives to the single distributions such as mean, standard deviation and
    median.

    In the nearest neighbour approach the function identify all the clusters and their contours, then calculates for each clusters the minimum distance 
    between every single pixel of one contours and with each pixel of the other contours. Then it keeps only one (or three) distances for particle. 
    Finally, it plots both the distribution and the relative metrics (both for single and all frames together) and an image for displaying the distances.

    Parameters:
    -----------
    @None

    Output:
    -----------
    @None
    """

    # Creating the image stack, then selecting the frames
    stack_creation()
    frames = frame_selection(subfolder_stack)
    rf = calculate_rescaling_factor(os.path.join(subfolder_stack, frames[0])) # Calculating the rescaling factor
    
    results_path = os.path.join(folder_path, "Results").replace("\\", "/") # Creating a folder for the results
    os.makedirs(results_path)

    # Initializing some variables
    data_set = []
    nn1 = []
    nn3 = []
    mean = []
    std_dev = []
    median = []
    mean_1 = []
    std_1 = []
    mean_3 = []
    std_3 = []
    flag_iter = True
    
    for filename in os.listdir(subfolder_stack): # Iterating over all the elements of the stack but analysing only the four selected
        if filename in frames:
            if flag_iter:
                data_processed, gs_image = image_processing(os.path.join(subfolder_stack, filename), True) # Processing the image
                flag_iter = False
            else:
                data_processed, gs_image = image_processing(os.path.join(subfolder_stack, filename), False) # Processing the image
                
            d = distance_calculation(data_processed, rf, results_path, filename) # Calculating the euclidean distance and ridge map
            mn = np.mean(d)
            data_set.append(d) # Saving the spacings and all the metrics 
            mean.append(mn)
            std_dev.append(np.std(d))
            median.append(np.median(d))

            # Starting nearest neighbour approach
            d1, d3 = nearest_neighbour_approach(data_processed, gs_image, mn, results_path, rf, filename)
            nn1.append(d1)
            nn3.append(d3)

            # Appending also the metrics
            mean_1.append(np.mean(d1))
            std_1.append(np.std(d1))

            mean_3.append(np.mean(d3))
            std_3.append(np.std(d3))
            

    total_distribution_plot(data_set, results_path, frames) # Plotting the final ED graph
    plotting_nearest_neighbour(nn1, nn3, frames, results_path)

    with open(os.path.join(results_path, "Results.csv"), 'w', newline = '') as csvfile: # Creating excel file
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["File name", "Mean (nm)", "Std dev (nm)", "Median (nm)", "", "1NN mean (nm)", "1NN std (nm)", "3NN mean (nm)", "3NN std (nm)"]) # Inserting all the metrics
            for fname, m, std, med, m1, std1, m3, std3 in zip(frames, mean, std_dev, median, mean_1, std_1, mean_3, std_3):
                csv_writer.writerow([fname, m, std, med, "", m1, std1, m3, std3])
