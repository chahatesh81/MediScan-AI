from __future__ import annotations

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = 224

MIN_FOREGROUND_AREA_RATIO = 0.25

CROP_MARGIN_RATIO = 0.03


def ensure_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert an image to single-channel grayscale.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if image.ndim == 2:
        grayscale = image

    elif image.ndim == 3:
        channels = image.shape[-1]

        if channels == 1:
            grayscale = image[..., 0]

        elif channels == 3:
            grayscale = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        elif channels == 4:
            grayscale = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )

        else:
            raise ValueError(
                "Unsupported channel count: "
                f"{channels}"
            )

    else:
        raise ValueError(
            "Expected a 2D or 3D image. "
            f"Received shape: {image.shape}"
        )

    return grayscale


def normalize_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize arbitrary numeric image data to uint8.
    """

    image = np.asarray(
        image
    )

    if image.dtype == np.uint8:
        return image.copy()

    image = image.astype(
        np.float32
    )

    finite_mask = np.isfinite(
        image
    )

    if not finite_mask.any():
        raise ValueError(
            "Image contains no finite values."
        )

    finite_values = image[
        finite_mask
    ]

    minimum = float(
        finite_values.min()
    )

    maximum = float(
        finite_values.max()
    )

    if maximum <= minimum:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        image - minimum
    ) / (
        maximum - minimum
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).astype(np.uint8)


def estimate_background_intensity(
    image: np.ndarray,
) -> float:
    """
    Estimate border/background intensity from image corners.
    """

    height, width = image.shape

    patch_height = max(
        2,
        int(height * 0.05),
    )

    patch_width = max(
        2,
        int(width * 0.05),
    )

    corner_pixels = np.concatenate(
        [
            image[
                :patch_height,
                :patch_width,
            ].ravel(),
            image[
                :patch_height,
                width - patch_width :,
            ].ravel(),
            image[
                height - patch_height :,
                :patch_width,
            ].ravel(),
            image[
                height - patch_height :,
                width - patch_width :,
            ].ravel(),
        ]
    )

    return float(
        np.median(corner_pixels)
    )


def build_foreground_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect the radiographic foreground.

    This is not lung segmentation.
    It only attempts to remove obvious empty padding
    and uniform image borders.
    """

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0,
    )

    background = (
        estimate_background_intensity(
            blurred
        )
    )

    difference = cv2.absdiff(
        blurred,
        np.full_like(
            blurred,
            int(round(background)),
        ),
    )

    otsu_threshold, _ = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    threshold = max(
        5.0,
        float(otsu_threshold) * 0.50,
    )

    mask = (
        difference > threshold
    ).astype(np.uint8) * 255

    kernel_size = max(
        3,
        int(
            round(
                min(image.shape)
                * 0.015
            )
        ),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


def find_foreground_bbox(
    image: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Return:
        x_min, y_min, x_max, y_max

    If detection is unsafe, return the full image.
    """

    height, width = image.shape

    full_bbox = (
        0,
        0,
        width,
        height,
    )

    mask = build_foreground_mask(
        image
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return full_bbox

    valid_contours = []

    minimum_component_area = (
        height
        * width
        * 0.005
    )

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area >= minimum_component_area:
            valid_contours.append(
                contour
            )

    if not valid_contours:
        return full_bbox

    points = np.concatenate(
        valid_contours,
        axis=0,
    )

    x, y, box_width, box_height = (
        cv2.boundingRect(points)
    )

    detected_area_ratio = (
        box_width
        * box_height
    ) / (
        width
        * height
    )

    if (
        detected_area_ratio
        < MIN_FOREGROUND_AREA_RATIO
    ):
        return full_bbox

    margin_x = int(
        width
        * CROP_MARGIN_RATIO
    )

    margin_y = int(
        height
        * CROP_MARGIN_RATIO
    )

    x_min = max(
        0,
        x - margin_x,
    )

    y_min = max(
        0,
        y - margin_y,
    )

    x_max = min(
        width,
        x + box_width + margin_x,
    )

    y_max = min(
        height,
        y + box_height + margin_y,
    )

    final_width = (
        x_max - x_min
    )

    final_height = (
        y_max - y_min
    )

    if (
        final_width <= 0
        or final_height <= 0
    ):
        return full_bbox

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def crop_foreground(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int],
]:
    """
    Crop obvious empty peripheral regions.
    """

    grayscale = ensure_grayscale(
        image
    )

    grayscale = normalize_to_uint8(
        grayscale
    )

    bbox = find_foreground_bbox(
        grayscale
    )

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    cropped = grayscale[
        y_min:y_max,
        x_min:x_max,
    ]

    if cropped.size == 0:
        return grayscale, (
            0,
            0,
            grayscale.shape[1],
            grayscale.shape[0],
        )

    return cropped, bbox


def resize_with_padding(
    image: np.ndarray,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize while preserving aspect ratio.

    Empty space is filled using the median border
    intensity of the cropped radiograph.
    """

    if target_size <= 0:
        raise ValueError(
            "target_size must be positive."
        )

    height, width = image.shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    scale = min(
        target_size / width,
        target_size / height,
    )

    resized_width = max(
        1,
        int(round(width * scale)),
    )

    resized_height = max(
        1,
        int(round(height * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=interpolation,
    )

    fill_value = int(
        round(
            estimate_background_intensity(
                image
            )
        )
    )

    canvas = np.full(
        (
            target_size,
            target_size,
        ),
        fill_value,
        dtype=np.uint8,
    )

    offset_x = (
        target_size
        - resized_width
    ) // 2

    offset_y = (
        target_size
        - resized_height
    ) // 2

    canvas[
        offset_y : (
            offset_y
            + resized_height
        ),
        offset_x : (
            offset_x
            + resized_width
        ),
    ] = resized

    return canvas


def preprocess_xray(
    image: np.ndarray,
    target_size: int = DEFAULT_IMAGE_SIZE,
    return_metadata: bool = False,
) -> (
    np.ndarray
    | tuple[
        np.ndarray,
        dict,
    ]
):
    """
    Full artifact-aware preprocessing pipeline.

    Output:
        RGB uint8 image with shape:
        (target_size, target_size, 3)
    """

    grayscale = ensure_grayscale(
        image
    )

    grayscale = normalize_to_uint8(
        grayscale
    )

    original_height, original_width = (
        grayscale.shape
    )

    cropped, bbox = crop_foreground(
        grayscale
    )

    processed_gray = resize_with_padding(
        cropped,
        target_size=target_size,
    )

    processed_rgb = cv2.cvtColor(
        processed_gray,
        cv2.COLOR_GRAY2RGB,
    )

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    original_area = (
        original_width
        * original_height
    )

    crop_area = (
        (x_max - x_min)
        * (y_max - y_min)
    )

    retained_area_ratio = (
        crop_area / original_area
    )

    metadata = {
        "original_width": (
            original_width
        ),
        "original_height": (
            original_height
        ),
        "crop_bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "cropped_width": (
            cropped.shape[1]
        ),
        "cropped_height": (
            cropped.shape[0]
        ),
        "retained_area_ratio": float(
            retained_area_ratio
        ),
        "target_size": target_size,
    }

    if return_metadata:
        return (
            processed_rgb,
            metadata,
        )

    return processed_rgb