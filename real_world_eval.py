import os
import numpy as np
import difflib

def evaluate_ocr_folder(folder="real_captcha_samples", debug=True):
    """
    Evaluate OCR performance on CAPTCHA images using fuzzy matching.
    Returns:
        avg_accuracy, max_accuracy, samples_tested, status
    """
                                                                
    # =========================
    # 1. Safe Imports
    # =========================
    try:
        import pytesseract               # OCR engine
        from PIL import Image            # Image processing
    except Exception as e:
        # If OCR libraries are not available, return default result
        return {
            "avg_accuracy": 0.0,
            "max_accuracy": 0.0,
            "samples_tested": 0,
            "status": f"OCR_NOT_AVAILABLE: {str(e)}"
        }

    # =========================
    # 2. Folder Check
    # =========================
    # Check if the folder exists
    if not os.path.exists(folder):
        return {
            "avg_accuracy": 0.0,
            "max_accuracy": 0.0,
            "samples_tested": 0,
            "status": "NO_FOLDER"
        }

    # Get all image files (png, jpg, jpeg)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Debug: print number of files found
    if debug:
        print(f"Total files found: {len(files)}")

    # If no images found
    if len(files) == 0:
        return {
            "avg_accuracy": 0.0,
            "max_accuracy": 0.0,
            "samples_tested": 0,
            "status": "NO_IMAGES"
        }

    # =========================
    # 3. Fuzzy Matching Function
    # =========================
    # Computes similarity score between predicted text and ground truth
    def char_match_score(pred, gt):
        # Clean and normalize text (remove spaces, uppercase)
        pred = pred.strip().upper()
        gt = gt.strip().upper()

        # If ground truth is empty, return 0
        if len(gt) == 0:
            return 0.0

        # Use sequence matcher to compute similarity (0 to 1)
        return difflib.SequenceMatcher(None, pred, gt).ratio()

    # =========================
    # 4. Process Images
    # =========================
    results = []   # Store accuracy scores for each image

    for file in files:
        path = os.path.join(folder, file)

        # Extract ground truth (GT) text from filename
        # Example: captcha_AB12C.png → GT = AB12C
        try:
            gt = file.split("_")[1].split(".")[0]
        except:
            gt = ""

        try:
            # Open image
            img = Image.open(path)

            # Run OCR to extract text from image
            text = pytesseract.image_to_string(img)

            # =========================
            # CLEAN TEXT
            # =========================
            # Remove newline and spaces for better comparison
            text = text.strip().replace("\n", "").replace(" ", "")

            # Compute similarity score with ground truth
            score = char_match_score(text, gt)

            # Store score
            results.append(score)

        except Exception as e:
            # If any error occurs while processing image
            if debug:
                print(f"[DEBUG] Error processing {file}: {e}")

            # Assign 0 score for failed case
            results.append(0.0)

    # =========================
    # 5. Final Metrics
    # =========================
    # If no valid results
    if len(results) == 0:
        return {
            "avg_accuracy": 0.0,
            "max_accuracy": 0.0,
            "samples_tested": 0,
            "status": "NO_VALID_RESULTS"
        }

    # Compute final evaluation metrics
    return {
        "avg_accuracy": float(np.mean(results)),   # average OCR accuracy
        "max_accuracy": float(np.max(results)),    # best-case OCR accuracy
        "samples_tested": len(results),            # number of images tested
        "status": "SUCCESS"
    }