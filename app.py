"""
Product Defect Detection System - Flask Web Application
REST API Backend with HTML/CSS/JavaScript Frontend
"""
import os
import sys
import time
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *
from src.predictor import DefectPredictor

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = OUTPUTS_DIR

# Ensure directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# Global predictor instance (lazy loaded)
predictor = None

def get_predictor():
    """Get or initialize the predictor instance"""
    global predictor
    if predictor is None:
        model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            predictor = DefectPredictor(model_path)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
    return predictor


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze uploaded image for defects
    
    Returns JSON:
    {
        "success": true/false,
        "is_defective": true/false,
        "predicted_class": "scratch",
        "confidence": 0.927,
        "severity_score": 78,
        "severity_level": "HIGH",
        "heatmap_url": "/outputs/heatmap_xxx.jpg",
        "inference_time_ms": 135,
        "all_probabilities": {...}
    }
    """
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[-1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'Invalid file type. Allowed: {allowed_extensions}'}), 400
        
        # Save uploaded file
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{unique_id}.{file_ext}"
        filepath = os.path.join(OUTPUTS_DIR, filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        
        # Run prediction with timing
        start_time = time.time()
        result = pred.predict_single(
            filepath,
            include_heatmap=True,
            include_severity=True,
            data_dir=os.path.join(BASE_DIR, 'data', 'train')
        )
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # Determine if defective
        predicted_class = result['predicted_class'].lower()
        is_defective = predicted_class != 'no_defect' and predicted_class != 'good'
        
        # Get heatmap URL
        heatmap_url = None
        if result.get('heatmap_path') and os.path.exists(result['heatmap_path']):
            heatmap_filename = os.path.basename(result['heatmap_path'])
            heatmap_url = f"/outputs/{heatmap_filename}"
        
        # Build response
        response = {
            'success': True,
            'is_defective': is_defective,
            'predicted_class': result['predicted_class'],
            'confidence': round(result['confidence'] * 100, 1),
            'severity_score': int(result.get('severity_score', 0) * 10),  # Convert 0-10 to 0-100
            'severity_level': result.get('severity_level', 'UNKNOWN'),
            'heatmap_url': heatmap_url,
            'original_url': f"/outputs/{filename}",
            'inference_time_ms': inference_time_ms,
            'all_probabilities': {k: round(v * 100, 1) for k, v in result.get('all_probabilities', {}).items()}
        }
        
        return jsonify(response)
        
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve files from outputs directory (heatmaps, uploaded images)"""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'model_path': model_path,
        'model_exists': os.path.exists(model_path)
    })


@app.route('/api/download-report', methods=['POST'])
def download_report():
    """
    Generate and download PDF report for analysis results
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Import report generator
        from src.report_generator import ReportGenerator
        from flask import send_file
        
        # Build predictions list from all_probabilities
        all_probs = data.get('all_probabilities', {})
        predictions = []
        for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            predictions.append({
                'class': class_name,
                'probability': prob / 100  # Convert back to 0-1
            })
        
        # Prepare result data for report
        result = {
            'predicted_class': data.get('predicted_class', 'unknown'),
            'confidence': data.get('confidence', 0) / 100,  # Convert back to 0-1
            'severity_score': data.get('severity_score', 0) / 10,  # Convert back to 0-10
            'severity_level': data.get('severity_level', 'UNKNOWN'),
            'all_probabilities': {k: v / 100 for k, v in all_probs.items()},
            'is_defective': data.get('is_defective', False),
            'inference_time_ms': data.get('inference_time_ms', 0),
            'predictions': predictions  # Required by report generator
        }
        
        # Get image paths
        original_url = data.get('original_url', '')
        heatmap_url = data.get('heatmap_url', '')
        
        original_path = os.path.join(OUTPUTS_DIR, os.path.basename(original_url)) if original_url else None
        heatmap_path = os.path.join(OUTPUTS_DIR, os.path.basename(heatmap_url)) if heatmap_url else None
        
        # Generate report
        generator = ReportGenerator()
        pdf_path = generator.generate_single_image_report(
            result,
            original_path,
            heatmap_path
        )
        
        # Send the file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"defect_report_{os.path.basename(original_path or 'analysis')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔍 Product Defect Detection System")
    print("="*60)
    print(f"Model path: {CHECKPOINT_PATH}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print("="*60)
    print("\n🚀 Starting Flask server...")
    print("   Open http://localhost:5000 in your browser\n")
    
    # Run Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)
