"""Generate synthetic vision dataset for testing."""

import numpy as np
from PIL import Image
from pathlib import Path

def create_synthetic_images():
    """Create simple synthetic images for testing."""
    output_dir = Path("data/vision")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image properties
    img_size = 224
    num_images = 100
    
    for i in range(num_images):
        # Create blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Add random shapes
        num_shapes = np.random.randint(1, 5)
        
        for j in range(num_shapes):
            shape_type = np.random.choice(['rectangle', 'circle'])
            color = np.random.randint(50, 255, size=3)
            
            if shape_type == 'rectangle':
                x1 = np.random.randint(0, img_size - 50)
                y1 = np.random.randint(0, img_size - 50)
                w = np.random.randint(20, 60)
                h = np.random.randint(20, 60)
                img[y1:y1+h, x1:x1+w] = color
            
            else:  # circle
                cx = np.random.randint(30, img_size - 30)
                cy = np.random.randint(30, img_size - 30)
                r = np.random.randint(10, 30)
                
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                img[mask] = color
        
        # Save image
        pil_img = Image.fromarray(img)
        pil_img.save(output_dir / f"image_{i:04d}.png")
    
    print(f"Created {num_images} synthetic images in {output_dir}")

if __name__ == "__main__":
    create_synthetic_images()
