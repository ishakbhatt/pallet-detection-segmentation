from PIL import Image, ImageEnhance
import shutil
import numpy as np
import os
import cv2
import random

class ImageAugmentation:
    
    def __init__(self):
        self.size = (650, 650)

    @staticmethod
    def _copy_label(label_path, augmentation_method):
        label_dir, label_filename = os.path.split(label_path)
        augment_label_filename = os.path.splitext(label_filename)[0] + augmentation_method + os.path.splitext(label_filename)[1]
        shutil.copy(label_path, os.path.join(label_dir, augment_label_filename))
        return augment_label_filename

    def _convert_to_pil(self, image):
        # Convert numpy array (BGR) to PIL Image (RGB)
        if isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    def _convert_to_cv(self, image):
        # Convert PIL Image (RGB) to numpy array (BGR)
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image

    def _brighten(self, image, label_path, factor):
        if factor <= 1.0:
            raise ValueError("Brightness factor must be > 1.0")
        image_pil = self._convert_to_pil(image)
        enhancer = ImageEnhance.Brightness(image_pil)
        brightened = enhancer.enhance(factor)
        return self._convert_to_cv(brightened)

    def _darken(self, image, label_path, factor):
        if factor >= 1.0:
            raise ValueError("Darkening factor must be < 1.0")
        image_pil = self._convert_to_pil(image)
        enhancer = ImageEnhance.Brightness(image_pil)
        darkened = enhancer.enhance(factor)
        return self._convert_to_cv(darkened)

    def _contrast(self, image, label_path, factor):
        self._copy_label(label_path, "_CONTRAST")
        image_pil = self._convert_to_pil(image)
        enhancer = ImageEnhance.Contrast(image_pil)
        contrasted = enhancer.enhance(factor)
        return self._convert_to_cv(contrasted)

    def _blur(self, image, label_path, kernel_size=(3, 3), sigma=0):
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Kernel size must be odd (e.g., (3, 3), (5, 5))")
        self._copy_label(label_path, "_BLUR")
        img_np = np.array(image)
        blurred = cv2.GaussianBlur(img_np, ksize=kernel_size, sigmaX=sigma)
        return blurred

    def _saturate(self, image, label_path, factor):
        self._copy_label(label_path, "_SATURATED")
        image_pil = self._convert_to_pil(image)
        enhancer = ImageEnhance.Color(image_pil)
        saturated = enhancer.enhance(factor)
        return self._convert_to_cv(saturated)

    def _save_augmented_image(self, image, image_path, suffix):
        dir_path, img_filename = os.path.split(image_path)
        name, ext = os.path.splitext(img_filename)
        new_filename = f"{name}{suffix}{ext}"
        new_path = os.path.join(dir_path, new_filename)

        if isinstance(image, np.ndarray):
            cv2.imwrite(new_path, image)
        else:
            image.save(new_path)

        return new_path

    def augment(self, image, image_path, label_path):
        augmented_image_paths = []
        augmented_label_paths = []
        if random.random() < 1:
            brightened = self._brighten(image, label_path, 1.2)
            augmented_label_paths.append(self._copy_label(label_path, "_BRIGHT"))
            augmented_image_paths.append(self._save_augmented_image(brightened, image_path, "_BRIGHT"))

        if random.random() < 1:
            darkened = self._darken(image, label_path, 0.8)
            augmented_label_paths.append(self._copy_label(label_path, "_DARK"))
            augmented_image_paths.append(self._save_augmented_image(darkened, image_path, "_DARK"))

        if random.random() < 1:
            contrasted = self._contrast(image, label_path, 1.2)
            augmented_label_paths.append(self._copy_label(label_path, "_CONTRAST"))
            augmented_image_paths.append(self._save_augmented_image(contrasted, image_path, "_CONTRAST"))

        if random.random() < 1:
            blurred = self._blur(image, label_path, (3, 3))
            augmented_label_paths.append(self._copy_label(label_path, "_BLUR"))
            augmented_image_paths.append(self._save_augmented_image(blurred, image_path, "_BLUR"))

        if random.random() < 1:
            saturated = self._saturate(image, label_path, 1.2)
            augmented_label_paths.append(self._copy_label(label_path, "_SATURATED"))
            augmented_image_paths.append(self._save_augmented_image(saturated, image_path, "_SATURATED"))

        return augmented_image_paths, augmented_label_paths

    def augment_and_shuffle_data(self, images_dir, labels_dir):
        images = os.listdir(images_dir)
        labels = os.listdir(labels_dir)

        augmented_images = []
        augmented_labels = []

        for img_filename, label_filename in zip(images, labels):
            img_path = os.path.join(images_dir, img_filename)
            label_path = os.path.join(labels_dir, label_filename)
            
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Failed to load image {img_path}, skipping.")
                continue
            
            augmented_image_paths, augmented_label_paths = self.augment(image, img_path, label_path)

            for augmented_image_path in augmented_image_paths:
                augmented_images.append(augmented_image_path)
                augmented_labels.append(label_path)

        augmented_data_pairs = list(zip(augmented_images, augmented_labels))

        random.shuffle(augmented_data_pairs)

        shuffled_augmented_images = [pair[0] for pair in augmented_data_pairs]
        shuffled_augmented_labels = [pair[1] for pair in augmented_data_pairs]

        return shuffled_augmented_images, shuffled_augmented_labels