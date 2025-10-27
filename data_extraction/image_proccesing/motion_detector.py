import cv2
import numpy as np
import argparse
from collections import deque
import json
import os
from pathlib import Path


class MotionDetector:
    def __init__(self, window_size: int = 15, threshold: float = 30.0):
        self.window_size = window_size
        self.threshold = threshold
        self.score_buffer = deque(maxlen=window_size)
        self.frame_count = 0
        self.previous_frame = None
        
    def calculate_motion_score(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> float:
        if previous_frame is None:
            return 0.0
            
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            
        if len(previous_frame.shape) == 3:
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        else:
            previous_gray = previous_frame
        
        if current_gray.shape != previous_gray.shape:
            current_gray = cv2.resize(current_gray, (previous_gray.shape[1], previous_gray.shape[0]))
        
        diff = np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
        return float(np.sum(diff) / diff.size)
    
    def update_buffer(self, score: float) -> None:
        self.score_buffer.append(score)
    
    def get_average_score(self) -> float:
        if not self.score_buffer:
            return 0.0
        return float(sum(self.score_buffer) / len(self.score_buffer))
    
    def has_significant_motion(self) -> bool:
        return self.get_average_score() >= self.threshold
    
    def process_frame(self, frame: np.ndarray) -> dict:
        motion_score = 0.0
        
        if self.previous_frame is not None:
            motion_score = self.calculate_motion_score(frame, self.previous_frame)
        
        self.update_buffer(motion_score)
        avg_score = self.get_average_score()
        has_motion = self.has_significant_motion()
        
        self.previous_frame = frame.copy()
        self.frame_count += 1
        
        return {
            'frame_number': self.frame_count,
            'motion_score': float(motion_score),
            'average_score': float(avg_score),
            'has_motion': has_motion
        }
    
    def reset(self):
        self.score_buffer.clear()
        self.frame_count = 0
        self.previous_frame = None


class VideoMotionAnalyzer:
    def __init__(self, window_size: int = 5, threshold: float = 100.0):
        self.detector = MotionDetector(window_size, threshold)
        
    def analyze_video(self, video_path: str, output_path: str = None, save_frames: str = None) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'No se pudo abrir: {video_path}'}
        
        results = []
        motion_frames = []
        no_motion_frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Crear carpeta para frames sin movimiento si se especifica
        if save_frames:
            output_dir = Path(save_frames)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            result = self.detector.process_frame(frame)
            results.append(result)
            
            if result['has_motion']:
                motion_frames.append({
                    'frame_number': result['frame_number'],
                    'timestamp': frame_idx / fps,
                    'motion_score': result['motion_score'],
                    'average_score': result['average_score']
                })
            else:
                no_motion_frames.append({
                    'frame_number': result['frame_number'],
                    'timestamp': frame_idx / fps,
                    'motion_score': result['motion_score'],
                    'average_score': result['average_score']
                })
                
                # Guardar frame sin movimiento
                if save_frames:
                    frame_filename = f"frame_{result['frame_number']:06d}_t{frame_idx/fps:.2f}s.jpg"
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
            
            frame_idx += 1
        
        cap.release()
        
        analysis_result = {
            'video_path': video_path,
            'total_frames': total_frames,
            'motion_frames_count': len(motion_frames),
            'no_motion_frames_count': len(no_motion_frames),
            'motion_ratio': len(motion_frames) / total_frames if total_frames > 0 else 0,
            'motion_frames': motion_frames,
            'no_motion_frames': no_motion_frames,
            'settings': {
                'window_size': self.detector.window_size,
                'threshold': self.detector.threshold
            }
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        return analysis_result
    
    def process_video_frames(self, video_path: str, callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            result = self.detector.process_frame(frame)
            results.append(result)
            
            if callback:
                callback(frame, result)
        
        cap.release()
        return results


def main():
    parser = argparse.ArgumentParser(description="Detector de movimiento")
    parser.add_argument('-v', '--video', type=str, required=True, help='Video')
    parser.add_argument('-w', '--window', type=int, default=15, help='Ventana temporal')
    parser.add_argument('-t', '--threshold', type=float, default=30.0, help='Umbral')
    parser.add_argument('-o', '--output', type=str, help='Archivo salida')
    parser.add_argument('--save-frames', type=str, help='Carpeta para guardar frames sin movimiento')
    parser.add_argument('--show-frames', action='store_true', help='Mostrar frames')
    
    args = parser.parse_args()
    
    analyzer = VideoMotionAnalyzer(args.window, args.threshold)
    
    if args.show_frames:
        def show_callback(frame, result):
            if result['has_motion']:
                cv2.imshow('Motion', frame)
                cv2.waitKey(1)
        
        analyzer.process_video_frames(args.video, show_callback)
        cv2.destroyAllWindows()
    else:
        results = analyzer.analyze_video(args.video, args.output, args.save_frames)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Frames: {results['total_frames']} | Con movimiento: {results['motion_frames_count']} | Sin movimiento: {results['no_motion_frames_count']}")
            if args.save_frames:
                print(f"Frames sin movimiento guardados en: {args.save_frames}")


if __name__ == "__main__":
    main()
