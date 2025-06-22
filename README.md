# From-Scratch Seam Carving for Content-Aware Object Removal

This project is a testament to first-principles problem-solving, **implementing a content-aware image resizing algorithm entirely from scratch based solely on a high-level problem description**. The resulting implementation remarkably converges with the fundamental principles of the renowned Seam Carving algorithm, **independently demonstrating the logic formalized in academic literature**. The core logic is built purely with Python and NumPy, showcasing a deep, practical understanding of computer vision fundamentals.

---

## Core Algorithm: Seam Carving

Seam Carving is not a simple pixel deletion. It's an advanced content-aware process that removes paths of pixels (seams) with the lowest energy, preserving the most important content of the image. This project implements the full pipeline:

1.  **Energy Map Calculation (`compute_energy`)**: The significance of each pixel is quantified by calculating an "energy" value. This is derived from the image's gradients. High-energy pixels correspond to edges and detailed textures, while low-energy pixels represent smooth areas.

2.  **Optimal Seam Identification (`find_vertical_path`)**: Using **dynamic programming**, the algorithm computes a cumulative energy matrix. This allows for the efficient identification of the "least important" connected path of pixels from the top of the image to the bottom—the seam.

3.  **Iterative Object Removal (`iterative_object_removal`)**: To remove a specific object, a binary mask is provided. The energy of the pixels within the mask is artificially set to zero, forcing the algorithm to prioritize carving seams that pass through the target object. This process is repeated until the entire object is removed.

4.  **Image Restoration (`restore_image_to_original_size`)**: After the object is carved out, the image is smaller. To restore it to its original dimensions, the algorithm reverses the process: it finds the lowest-energy seams in the modified image and duplicates them.

---

## Project Structure & Files

The project is structured with a clear separation of concerns, making it clean and understandable.

```
From-Scratch-Seam-Carving_for_Content-Aware-Object-Removal/
├── main.py                  # The main executable script to run the project.
├── segmentation.py          # A library containing all core Seam Carving functions.
├── requirements.txt         # Python dependencies.
├── input/                   # Directory for input images and masks.
│   ├── ball.jpg
│   ├── ball_mask.jpg
│   ├── chick.jpg
│   ├── chick_mask.jpg
│   ├── object.jpg
│   └── object_mask.jpg
└── output/                  # Directory for all generated results.
```

- **`main.py`**: This is the **entry point** of the project. It handles loading the sample data from the `input` folder, orchestrates the calls to the functions in `segmentation.py`, and saves the final results and comparison images to the `output` folder. **To run the project, you only need to run this file.**
- **`segmentation.py`**: This file acts as a **module/library**. It contains all the core, from-scratch algorithmic logic for Seam Carving (energy calculation, path finding, seam removal/insertion). It does not produce any output on its own; it only provides the tools for `main.py` to use.

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EonTechie/From-Scratch-Seam-Carving_for_Content-Aware-Object-Removal.git
    cd From-Scratch-Seam-Carving_for_Content-Aware-Object-Removal
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution

To run the entire object removal pipeline, execute the `main.py` script from the project's root directory. The script will automatically process all sample images.

```bash
python main.py
```

---

## Expected Output

After running the script, the `output` directory will be created and populated with the following files:

```
output/
├── ball_final.jpg
├── ball_removal_comparison.jpg
├── chick_final.jpg
├── chick_removal_comparison.jpg
├── object_final.jpg
└── object_removal_comparison.jpg
```

---

## Implementation Highlights for Recruiters

This project serves as a strong indicator of core software engineering and algorithmic skills relevant to **AI/ML, Data Science, and Computer Vision** roles.

*   **Algorithmic Proficiency**: The implementation of a non-trivial algorithm from a research paper using fundamental tools demonstrates a deep understanding of **dynamic programming**, **optimization**, and **image processing fundamentals**.

*   **"From Scratch" Approach**: By intentionally avoiding one-line solutions like `cv2.inpaint()`, this project proves the ability to translate complex algorithmic concepts into functional, low-level code.

*   **Software Engineering Principles**: The code is structured with a clear **separation of concerns** (`main.py` for execution, `segmentation.py` for logic), which is a cornerstone of maintainable and scalable software.

*   **Performance-Conscious Coding**: The use of vectorized **NumPy** operations showcases an understanding of efficient computation, a critical skill in data processing and ML pipelines.


## Acknowledgments

*   This work was developed independently from a problem description and was later found to be a from-scratch re-implementation of the algorithm described in the paper **"Seam Carving for Content-Aware Image Resizing"** by Shai Avidan and Ariel Shamir.

## Contact

- **GitHub**: [@EonTechie](https://github.com/EonTechie)

--- 