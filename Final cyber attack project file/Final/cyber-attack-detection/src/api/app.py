"""
Flask REST API for Cyber Attack Detection System
Provides endpoints for real-time monitoring and analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models import CyberAttackDetector
from utils.data_generator import CyberAttackDataGenerator


app = Flask(__name__)
CORS(app)

# Global variables
detector = None
data_generator = CyberAttackDataGenerator()

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Cyber Attack Detection API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /train': 'Train ML models',
            'POST /predict': 'Make predictions',
            'POST /generate-data': 'Generate sample data',
            'GET /models/performance': 'Get model performance',
            'POST /demo/attack': 'Demonstrate attack detection',
            'POST /models/load': 'Load pre-trained models',
            'GET /stats/overview': 'Get system overview',
            'POST /predict/upload': 'Upload file and make predictions'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': detector is not None,
        'monitoring_active': False
    })

@app.route('/train', methods=['POST'])
def train_models():
    """Train machine learning models"""
    global detector
    
    try:
        data = request.get_json() or {}
        num_samples = data.get('num_samples', 5000)
        
        # Generate training data
        print(f"Generating {num_samples} samples for training...")
        training_data = data_generator.generate_complete_dataset(num_samples)
        
        # Initialize and train detector
        detector = CyberAttackDetector()
        detector.train_all_models(training_data)
        
        # Save models
        detector.save_models()
        
        # Get performance metrics
        performance = detector.get_model_performance()
        
        return jsonify({
            'message': 'Models trained successfully',
            'samples_used': len(training_data),
            'performance': performance,
            'best_model': detector.best_model_name
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on new data"""
    global detector
    
    if detector is None:
        return jsonify({
            'error': 'Models not trained. Please train models first.'
        }), 400
    
    try:
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame(data['data'])
        
        # Make predictions
        predictions = detector.predict(input_data)
        
        # Get prediction probabilities if available
        try:
            if hasattr(detector.best_model, 'predict_proba'):
                probabilities = detector.best_model.predict_proba(input_data)
                prob_dict = {}
                for i, pred in enumerate(predictions):
                    prob_dict[pred] = float(np.max(probabilities[i]))
            else:
                prob_dict = None
        except:
            prob_dict = None
        
        # Count predictions
        prediction_counts = pd.Series(predictions).value_counts().to_dict()
        
        return jsonify({
            'predictions': predictions.tolist(),
            'prediction_counts': prediction_counts,
            'probabilities': prob_dict,
            'model_used': detector.best_model_name,
            'total_samples': len(predictions)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500



@app.route('/generate-data', methods=['POST'])
def generate_sample_data():
    """Generate sample cyber attack data"""
    try:
        data = request.get_json() or {}
        num_samples = data.get('num_samples', 1000)
        
        # Generate data
        sample_data = data_generator.generate_complete_dataset(num_samples)
        
        # Convert to JSON-serializable format
        json_data = sample_data.to_dict('records')
        
        return jsonify({
            'message': f'Generated {num_samples} samples',
            'data': json_data,
            'attack_distribution': sample_data['attack_type'].value_counts().to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to generate data: {str(e)}'
        }), 500

@app.route('/models/performance')
def get_model_performance():
    """Get performance metrics for all models"""
    global detector
    
    if detector is None:
        return jsonify({
            'error': 'Models not trained'
        }), 400
    
    try:
        performance = detector.get_model_performance()
        return jsonify({
            'performance': performance,
            'best_model': detector.best_model_name
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get performance: {str(e)}'
        }), 500

@app.route('/demo/attack', methods=['POST'])
def demonstrate_attack():
    """Demonstrate attack detection with sample data"""
    global detector
    
    if detector is None:
        return jsonify({
            'error': 'Models not trained. Please train models first.'
        }), 400
    
    try:
        data = request.get_json() or {}
        attack_type = data.get('attack_type', 'ddos')
        num_samples = data.get('num_samples', 100)
        
        # Generate specific attack data
        if attack_type == 'ddos':
            attack_data = data_generator.generate_ddos_attack(num_samples)
        elif attack_type == 'sql_injection':
            attack_data = data_generator.generate_sql_injection_attack(num_samples)
        elif attack_type == 'xss':
            attack_data = data_generator.generate_xss_attack(num_samples)
        elif attack_type == 'brute_force':
            attack_data = data_generator.generate_brute_force_attack(num_samples)
        elif attack_type == 'port_scan':
            attack_data = data_generator.generate_port_scan_attack(num_samples)
        else:
            return jsonify({
                'error': f'Unknown attack type: {attack_type}'
            }), 400
        
        # Remove attack_type column for prediction
        attack_data_for_prediction = attack_data.drop(columns=['attack_type', 'attack_category'])
        if 'timestamp' in attack_data_for_prediction.columns:
            attack_data_for_prediction = attack_data_for_prediction.drop(columns=['timestamp'])
        
        # Make predictions
        predictions = detector.predict(attack_data_for_prediction)
        
        # Analyze results
        prediction_counts = pd.Series(predictions).value_counts()
        correct_predictions = prediction_counts.get(attack_type, 0)
        accuracy = (correct_predictions / len(predictions)) * 100
        
        return jsonify({
            'attack_type': attack_type,
            'samples_generated': len(attack_data),
            'predictions': predictions.tolist(),
            'prediction_counts': prediction_counts.to_dict(),
            'detection_accuracy': accuracy,
            'correct_predictions': int(correct_predictions),
            'model_used': detector.best_model_name
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Demo failed: {str(e)}'
        }), 500

@app.route('/models/load', methods=['POST'])
def load_models():
    """Load pre-trained models"""
    global detector
    
    try:
        detector = CyberAttackDetector()
        detector.load_models()
        
        return jsonify({
            'message': 'Models loaded successfully',
            'available_models': list(detector.models.keys()),
            'best_model': detector.best_model_name
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to load models: {str(e)}'
        }), 500

@app.route('/stats/overview')
def get_system_overview():
    """Get system overview statistics"""
    global detector
    
    try:
        stats = {
            'models_loaded': detector is not None,
            'monitoring_active': False,
            'timestamp': datetime.now().isoformat()
        }
        
        if detector:
            stats['available_models'] = list(detector.models.keys())
            stats['best_model'] = detector.best_model_name
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get overview: {str(e)}'
        }), 500

@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    """Upload file and make predictions"""
    global detector
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'csv', 'xlsx', 'xls'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'File type not supported. Please upload: {", ".join(allowed_extensions)}'}), 400
        
        # Load the data
        if file_extension == 'csv':
            data = pd.read_csv(file)
        else:
            # For Excel files
            data = pd.read_excel(file)
        
        # Check if required columns exist
        required_columns = [
            'duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes',
            'flag', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'required_columns': required_columns
            }), 400
        
        # Ensure detector is loaded
        if detector is None:
            try:
                detector = CyberAttackDetector()
                detector.load_models()
            except Exception as e:
                return jsonify({'error': 'Models not trained. Please train models first.'}), 400
        
        # Make predictions
        predictions = detector.predict(data)
        
        return jsonify({
            'message': 'File analyzed successfully',
            'filename': file.filename,
            'rows_analyzed': len(data),
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to analyze file: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Starting Cyber Attack Detection API...")
    print("Available endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  POST /train - Train ML models")
    print("  POST /predict - Make predictions")
    print("  POST /generate-data - Generate sample data")
    print("  GET  /models/performance - Get model performance")
    print("  POST /demo/attack - Demonstrate attack detection")
    print("  POST /models/load - Load pre-trained models")
    print("  GET  /stats/overview - Get system overview")
    print("  POST /predict/upload - Upload file and make predictions")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
