import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import List, Dict
import json


class DarkFrameDetector:
    def __init__(self, brightness_threshold: int = 30, dark_pixel_ratio: float = 0.5):
        self.brightness_threshold = brightness_threshold
        self.dark_pixel_ratio = dark_pixel_ratio
        
    def calculate_brightness_metrics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        total_pixels = gray.size
        dark_pixels = np.sum(gray < self.brightness_threshold)
        dark_ratio = dark_pixels / total_pixels
        
        return {
            'dark_ratio': float(dark_ratio),
            'mean_brightness': float(np.mean(gray)),
            'is_dark': dark_ratio >= self.dark_pixel_ratio
        }
    
    def analyze_image(self, image_path: str) -> Dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'No se pudo cargar: {image_path}', 'success': False}
            
            metrics = self.calculate_brightness_metrics(image)
            return {
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'success': True,
                **metrics
            }
            
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'error': str(e),
                'success': False
            }
    
    def analyze_directory(self, directory_path: str, supported_formats: List[str] = None, save_dark_images: str = None) -> Dict:
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        directory_path = Path(directory_path)
        if not directory_path.exists():
            return {'error': f'Directorio no existe: {directory_path}'}
        
        results = []
        dark_images = []
        
        # Crear carpeta para imágenes oscuras si se especifica
        if save_dark_images:
            output_dir = Path(save_dark_images)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in supported_formats:
            image_files.extend(directory_path.glob(f'*{ext}'))
            image_files.extend(directory_path.glob(f'*{ext.upper()}'))
        
        for image_file in image_files:
            result = self.analyze_image(str(image_file))
            results.append(result)
            
            if result.get('success', False) and result.get('is_dark', False):
                dark_images.append(result)
                
                # Guardar imagen oscura
                if save_dark_images:
                    # Leer imagen original
                    original_image = cv2.imread(str(image_file))
                    if original_image is not None:
                        # Crear nombre de archivo con información del análisis
                        filename = f"dark_{result['dark_ratio']:.2f}_{image_file.name}"
                        output_path = output_dir / filename
                        cv2.imwrite(str(output_path), original_image)
        
        return {
            'total_images': len(image_files),
            'dark_images_count': len(dark_images),
            'dark_images_ratio': float(len(dark_images) / len(image_files) if image_files else 0),
            'dark_images': dark_images,
            'all_results': results
        }
    
    def generate_report(self, analysis_results: Dict, output_path: str = None) -> str:
        if 'total_images' in analysis_results:
            stats = analysis_results
            report = f"Total: {stats['total_images']} | Oscuras: {stats['dark_images_count']} ({stats['dark_images_ratio']:.1%})"
            if stats['dark_images']:
                report += "\nImágenes oscuras:"
                for img in stats['dark_images']:
                    report += f"\n  - {img['filename']} ({img['dark_ratio']:.1%})"
        else:
            img = analysis_results
            report = f"{img['filename']}: {img['dark_ratio']:.1%} píxeles oscuros - {'OSCURA' if img['is_dark'] else 'NORMAL'}"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Detector de imágenes oscuras")
    parser.add_argument('-i', '--image', type=str, help='Imagen individual')
    parser.add_argument('-d', '--directory', type=str, help='Directorio con imágenes')
    parser.add_argument('-t', '--threshold', type=int, default=30, help='Umbral de brillo (0-255)')
    parser.add_argument('-r', '--ratio', type=float, default=0.8, help='Proporción mínima de píxeles oscuros (0-1)')
    parser.add_argument('-o', '--output', type=str, help='Archivo de salida')
    parser.add_argument('--save-dark', type=str, help='Carpeta para guardar imágenes oscuras')
    parser.add_argument('--json', action='store_true', help='Guardar en JSON')
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        parser.error("Especifica una imagen (-i) o directorio (-d)")
    
    detector = DarkFrameDetector(args.threshold, args.ratio)
    
    if args.image:
        results = detector.analyze_image(args.image)
        if results.get('success', False):
            report = detector.generate_report(results, args.output)
            print(report)
            
            # Guardar imagen oscura individual si es oscura
            if results.get('is_dark', False) and args.save_dark:
                output_dir = Path(args.save_dark)
                output_dir.mkdir(parents=True, exist_ok=True)
                original_image = cv2.imread(args.image)
                if original_image is not None:
                    filename = f"dark_{results['dark_ratio']:.2f}_{os.path.basename(args.image)}"
                    output_path = output_dir / filename
                    cv2.imwrite(str(output_path), original_image)
                    print(f"Imagen oscura guardada en: {output_path}")
        else:
            print(f"Error: {results.get('error')}")
    
    elif args.directory:
        results = detector.analyze_directory(args.directory, save_dark_images=args.save_dark)
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            report = detector.generate_report(results, args.output)
            print(report)
            if args.save_dark:
                print(f"Imágenes oscuras guardadas en: {args.save_dark}")
    
    if args.json and args.output:
        json_path = args.output.replace('.txt', '.json') if args.output.endswith('.txt') else args.output + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
