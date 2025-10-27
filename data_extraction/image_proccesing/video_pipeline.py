import cv2
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

from .motion_detector import VideoMotionAnalyzer
from .dark_frames_detector import DarkFrameDetector


class VideoProcessingPipeline:
    def __init__(self, motion_window=15, motion_threshold=30.0, 
                 brightness_threshold=30, dark_pixel_ratio=0.5, skip_frames_after_motion=15):
        self.motion_analyzer = VideoMotionAnalyzer(motion_window, motion_threshold)
        self.dark_detector = DarkFrameDetector(brightness_threshold, dark_pixel_ratio)
        self.skip_frames_after_motion = skip_frames_after_motion
        
    def process_video(self, video_path: str, output_dir: str = None) -> dict:
        video_name = Path(video_path).stem
        
        if output_dir is not None:
            video_output_dir = Path(output_dir) / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            no_motion_frames_dir = video_output_dir / "frames_sin_movimiento"
            dark_frames_dir = video_output_dir / "frames_oscuros"
            problematic_frames_dir = video_output_dir / "frames_problematicos"
            
            no_motion_frames_dir.mkdir(parents=True, exist_ok=True)
            dark_frames_dir.mkdir(parents=True, exist_ok=True)
            problematic_frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            no_motion_frames_dir = None
            dark_frames_dir = None
            problematic_frames_dir = None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'No se pudo abrir el video: {video_path}'}
        
        self.motion_analyzer.detector.reset()
        
        frame_count = 0
        motion_skip_counter = 0
        problematic_frames = []
        no_motion_frames = []
        dark_frames = []
        dark_frames_count = 0
        no_motion_frames_count = 0
        motion_frames_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            dark_metrics = self.dark_detector.calculate_brightness_metrics(frame)
            
            motion_result = self.motion_analyzer.detector.process_frame(frame)
            
            if motion_skip_counter > 0:
                motion_skip_counter -= 1
                
                if motion_result['has_motion']:
                    motion_skip_counter = self.skip_frames_after_motion
                
                if motion_skip_counter == 0:
                    self.motion_analyzer.detector.reset()
            elif motion_result['has_motion']:
                motion_skip_counter = self.skip_frames_after_motion
            
            
            if dark_metrics['is_dark']:
                dark_frames_count += 1
                dark_frames.append({
                    'frame_number': frame_count,
                    'timestamp': float(frame_count / cap.get(cv2.CAP_PROP_FPS)),
                    'dark_ratio': float(dark_metrics['dark_ratio']),
                    'mean_brightness': float(dark_metrics['mean_brightness'])
                })
                
                if dark_frames_dir is not None:
                    frame_filename = f"dark_{dark_metrics['dark_ratio']:.2f}_frame_{frame_count:06d}.jpg"
                    frame_path = dark_frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
            
            if motion_result['has_motion']:
                motion_frames_count += 1
            else:
                no_motion_frames_count += 1
                no_motion_frames.append({
                    'frame_number': frame_count,
                    'timestamp': float(frame_count / cap.get(cv2.CAP_PROP_FPS)),
                    'motion_score': float(motion_result['motion_score']),
                    'average_score': float(motion_result['average_score'])
                })
                
                if no_motion_frames_dir is not None:
                    frame_filename = f"no_motion_{motion_result['motion_score']:.2f}_frame_{frame_count:06d}.jpg"
                    frame_path = no_motion_frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
            
            if dark_metrics['is_dark'] and not motion_result['has_motion']:
                problematic_frames.append({
                    'frame_number': frame_count,
                    'timestamp': float(frame_count / cap.get(cv2.CAP_PROP_FPS)),
                    'dark_ratio': float(dark_metrics['dark_ratio']),
                    'motion_score': float(motion_result['motion_score'])
                })
                
                if problematic_frames_dir is not None:
                    frame_filename = f"problematic_{dark_metrics['dark_ratio']:.2f}_frame_{frame_count:06d}.jpg"
                    frame_path = problematic_frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
            
            frame_count += 1
        
        cap.release()
        
        
        results = {
            'video_path': video_path,
            'video_name': video_name,
            'total_frames': frame_count,
            'motion_analysis': {
                'frames_with_motion': motion_frames_count,
                'frames_without_motion': no_motion_frames_count,
                'motion_ratio': float(motion_frames_count / frame_count if frame_count > 0 else 0)
            },
            'dark_analysis': {
                'dark_frames_count': dark_frames_count,
                'dark_ratio': float(dark_frames_count / frame_count if frame_count > 0 else 0),
                'dark_frames': dark_frames
            },
            'no_motion_analysis': {
                'no_motion_frames_count': no_motion_frames_count,
                'no_motion_ratio': float(no_motion_frames_count / frame_count if frame_count > 0 else 0),
                'no_motion_frames': no_motion_frames
            },
            'problematic_frames': {
                'count': len(problematic_frames),
                'ratio': float(len(problematic_frames) / frame_count if frame_count > 0 else 0),
                'frames': problematic_frames
            },
            'output_directories': {
                'no_motion_frames': str(no_motion_frames_dir),
                'dark_frames': str(dark_frames_dir),
                'problematic_frames': str(problematic_frames_dir)
            },
            'settings': {
                'motion_window': self.motion_analyzer.detector.window_size,
                'motion_threshold': self.motion_analyzer.detector.threshold,
                'brightness_threshold': self.dark_detector.brightness_threshold,
                'dark_pixel_ratio': self.dark_detector.dark_pixel_ratio,
                'skip_frames_after_motion': self.skip_frames_after_motion
            }
        }
        
        if output_dir is not None:
            results_file = video_output_dir / f"{video_name}_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> dict:
        input_path = Path(input_dir)
        if not input_path.exists():
            return {'error': f'Directorio no existe: {input_dir}'}
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            return {'error': f'No se encontraron videos en: {input_dir}'}
        
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None
        
        all_results = []
        successful_videos = 0
        failed_videos = 0
        
        for video_file in video_files:
            try:
                result = self.process_video(str(video_file), str(output_path) if output_path is not None else None)
                if 'error' in result:
                    failed_videos += 1
                else:
                    successful_videos += 1
                
                all_results.append(result)
                
            except Exception as e:
                failed_videos += 1
                all_results.append({
                    'video_path': str(video_file),
                    'error': str(e)
                })
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(input_path),
            'output_directory': str(output_path) if output_path is not None else None,
            'total_videos': len(video_files),
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'video_results': all_results,
            'settings': {
                'motion_window': self.motion_analyzer.detector.window_size,
                'motion_threshold': self.motion_analyzer.detector.threshold,
                'brightness_threshold': self.dark_detector.brightness_threshold,
                'dark_pixel_ratio': self.dark_detector.dark_pixel_ratio,
                'skip_frames_after_motion': self.skip_frames_after_motion
            }
        }
        
        if output_path is not None:
            summary_file = output_path / "processing_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Pipeline de procesamiento de videos")
    parser.add_argument('-i', '--input', type=str, required=True, help='Directorio con videos')
    parser.add_argument('-o', '--output', type=str, required=True, help='Directorio de salida')
    parser.add_argument('-mw', '--motion-window', type=int, default=15, help='Ventana temporal para movimiento')
    parser.add_argument('-mt', '--motion-threshold', type=float, default=6.5, help='Umbral de movimiento')
    parser.add_argument('-bt', '--brightness-threshold', type=int, default=10, help='Umbral de brillo (0-255)')
    parser.add_argument('-dr', '--dark-ratio', type=float, default=0.8, help='Proporción mínima de píxeles oscuros')
    parser.add_argument('-sf', '--skip-frames', type=int, default=15, help='Frames a saltar después de detectar movimiento')
    
    args = parser.parse_args()
    
    pipeline = VideoProcessingPipeline(
        motion_window=args.motion_window,
        motion_threshold=args.motion_threshold,
        brightness_threshold=args.brightness_threshold,
        dark_pixel_ratio=args.dark_ratio,
        skip_frames_after_motion=args.skip_frames
    )
    
    results = pipeline.process_directory(args.input, args.output)
    
    if 'error' in results:
        sys.exit(1)
    
    total_dark_frames = 0
    total_no_motion_frames = 0
    total_problematic_frames = 0
    total_frames = 0
    
    for video_result in results['video_results']:
        if 'error' not in video_result:
            total_frames += video_result['total_frames']
            total_no_motion_frames += video_result['motion_analysis']['frames_without_motion']
            total_dark_frames += video_result['dark_analysis']['dark_frames_count']
            total_problematic_frames += video_result['problematic_frames']['count']
    


if __name__ == "__main__":
    main()
