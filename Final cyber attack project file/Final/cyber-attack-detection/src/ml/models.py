"""
Machine Learning Models for Cyber Attack Detection
Implements various ML algorithms for detecting cyber attacks
"""

import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class CyberAttackDetector:
    """Main class for cyber attack detection using multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        
    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'attack_type'):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = data.drop([target_column, 'attack_category', 'timestamp'], axis=1, errors='ignore')
        y = data[target_column]
        
        # Handle categorical variables
        categorical_columns = ['protocol_type', 'service', 'flag']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Convert to numpy arrays
        X = X.values.astype(float)
        y = y.values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=20)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        self.feature_selectors['main'] = selector
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.scalers['main'] = scaler
        
        print(f"Data preprocessed: {X_train_scaled.shape[0]} training samples, {X_test_scaled.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search for best parameters
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Train with best parameters
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.models['random_forest'] = best_rf
        self.model_performance['random_forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': grid_search.best_params_
        }
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return best_rf
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train Support Vector Machine model"""
        print("Training SVM...")
        
        # Use a subset for faster training
        if len(X_train) > 5000:
            indices = np.random.choice(len(X_train), 5000, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        
        # Grid search
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train_subset, y_train_subset)
        
        # Train with best parameters
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.models['svm'] = best_svm
        self.model_performance['svm'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': grid_search.best_params_
        }
        
        print(f"SVM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return best_svm
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        # Encode labels for neural network
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        self.label_encoders['target'] = le
        
        # Convert to one-hot encoding
        num_classes = len(le.classes_)
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes)
        
        # Build neural network
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train_onehot,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_decoded = le.inverse_transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_decoded)
        precision = precision_score(y_test, y_pred_decoded, average='weighted')
        recall = recall_score(y_test, y_pred_decoded, average='weighted')
        f1 = f1_score(y_test, y_pred_decoded, average='weighted')
        
        self.models['neural_network'] = model
        self.model_performance['neural_network'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'history': history.history
        }
        
        print(f"Neural Network - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5, 7]
        }
        
        # Grid search
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Train with best parameters
        best_gb = grid_search.best_estimator_
        y_pred = best_gb.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.models['gradient_boosting'] = best_gb
        self.model_performance['gradient_boosting'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': grid_search.best_params_
        }
        
        print(f"Gradient Boosting - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return best_gb
    
    def train_all_models(self, data: pd.DataFrame):
        """Train all machine learning models"""
        print("Starting model training...")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Train different models
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_neural_network(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # Find best model
        self.find_best_model()
        
        print("All models trained successfully!")
        return self.models
    
    def find_best_model(self):
        """Find the best performing model based on F1 score"""
        best_f1 = 0
        best_model_name = None
        
        for model_name, performance in self.model_performance.items():
            if performance['f1_score'] > best_f1:
                best_f1 = performance['f1_score']
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} with F1 score: {best_f1:.4f}")
    
    def predict(self, data: pd.DataFrame, model_name: str = None):
        """Make predictions using the specified model or best model"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        # Preprocess input data
        X = self.preprocess_input(data)
        
        # Make prediction
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred = self.label_encoders['target'].inverse_transform(y_pred)
        else:
            y_pred = model.predict(X)
        
        return y_pred
    
    def preprocess_input(self, data: pd.DataFrame):
        """Preprocess input data for prediction"""
        # Handle categorical variables
        X = data.copy()
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            if col in X.columns:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Convert to numpy array
        X = X.values.astype(float)
        
        # Apply feature selection
        X = self.feature_selectors['main'].transform(X)
        
        # Apply scaling
        X = self.scalers['main'].transform(X)
        
        return X
    
    def get_model_performance(self):
        """Get performance metrics for all models"""
        return self.model_performance
    
    def save_models(self, filepath: str = 'models/'):
        """Save all trained models"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f'{filepath}{name}_model.h5')
            else:
                joblib.dump(model, f'{filepath}{name}_model.pkl')
        
        # Save preprocessors
        joblib.dump(self.scalers, f'{filepath}scalers.pkl')
        joblib.dump(self.label_encoders, f'{filepath}label_encoders.pkl')
        joblib.dump(self.feature_selectors, f'{filepath}feature_selectors.pkl')
        
        # Save performance metrics
        with open(f'{filepath}model_performance.json', 'w') as f:
            import json
            # Convert numpy types to native Python types
            performance_dict = {}
            for key, value in self.model_performance.items():
                performance_dict[key] = {}
                for metric, val in value.items():
                    if isinstance(val, (np.integer, np.floating)):
                        performance_dict[key][metric] = val.item()
                    elif metric == 'history':
                        performance_dict[key][metric] = {
                            k: [float(x) for x in v] for k, v in val.items()
                        }
                    else:
                        performance_dict[key][metric] = val
            json.dump(performance_dict, f, indent=2)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = 'models/'):
        """Load trained models"""
        # Load models
        for name in ['random_forest', 'svm', 'gradient_boosting']:
            try:
                self.models[name] = joblib.load(f'{filepath}{name}_model.pkl')
            except:
                print(f"Could not load {name} model")
        
        # Load neural network
        try:
            self.models['neural_network'] = keras.models.load_model(f'{filepath}neural_network_model.h5')
        except:
            print("Could not load neural network model")
        
        # Load preprocessors
        self.scalers = joblib.load(f'{filepath}scalers.pkl')
        self.label_encoders = joblib.load(f'{filepath}label_encoders.pkl')
        self.feature_selectors = joblib.load(f'{filepath}feature_selectors.pkl')
        
        # Load performance metrics
        try:
            with open(f'{filepath}model_performance.json', 'r') as f:
                import json
                self.model_performance = json.load(f)
        except:
            print("Could not load performance metrics")
        
        # Find best model after loading
        self.find_best_model()
        
        print("Models loaded successfully")

if __name__ == "__main__":
    # Example usage
    from src.utils.data_generator import CyberAttackDataGenerator
    
    # Generate data
    generator = CyberAttackDataGenerator()
    data = generator.generate_complete_dataset(1000)
    
    # Train models
    detector = CyberAttackDetector()
    detector.train_all_models(data)
    
    # Save models
    detector.save_models()
