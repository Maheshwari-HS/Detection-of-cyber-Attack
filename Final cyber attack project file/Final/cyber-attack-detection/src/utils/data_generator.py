"""
Data Generator for Cyber Attack Detection
Creates realistic datasets for training and testing ML models
"""

import numpy as np
import pandas as pd
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

class CyberAttackDataGenerator:
    """Generates realistic cyber attack datasets for demonstration"""
    
    def __init__(self):
        self.attack_types = {
            'ddos': 'Distributed Denial of Service',
            'sql_injection': 'SQL Injection',
            'xss': 'Cross-Site Scripting',
            'brute_force': 'Brute Force Attack',
            'port_scan': 'Port Scanning',
            'normal': 'Normal Traffic'
        }
        
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
    
    def generate_normal_traffic(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate normal network traffic data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(100),
                'protocol_type': np.random.choice(['tcp', 'udp', 'icmp']),
                'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns']),
                'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'RSTR']),
                'src_bytes': np.random.poisson(1000),
                'dst_bytes': np.random.poisson(1000),
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 1,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(5),
                'srv_count': np.random.poisson(3),
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'rerror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': np.random.uniform(0.8, 1.0),
                'diff_srv_rate': np.random.uniform(0.0, 0.2),
                'srv_diff_host_rate': np.random.uniform(0.0, 0.1),
                'dst_host_count': np.random.poisson(10),
                'dst_host_srv_count': np.random.poisson(8),
                'dst_host_same_srv_rate': np.random.uniform(0.8, 1.0),
                'dst_host_diff_srv_rate': np.random.uniform(0.0, 0.2),
                'dst_host_same_src_port_rate': np.random.uniform(0.8, 1.0),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.0, 0.1),
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0,
                'attack_type': 'normal',
                'attack_category': 'normal'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_ddos_attack(self, num_samples: int = 500) -> pd.DataFrame:
        """Generate DDoS attack data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(10),
                'protocol_type': 'tcp',
                'service': np.random.choice(['http', 'ftp', 'smtp']),
                'flag': np.random.choice(['S0', 'REJ']),
                'src_bytes': np.random.poisson(100),
                'dst_bytes': 0,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(50),
                'srv_count': np.random.poisson(50),
                'serror_rate': np.random.uniform(0.8, 1.0),
                'srv_serror_rate': np.random.uniform(0.8, 1.0),
                'rerror_rate': np.random.uniform(0.0, 0.2),
                'srv_rerror_rate': np.random.uniform(0.0, 0.2),
                'same_srv_rate': np.random.uniform(0.0, 0.1),
                'diff_srv_rate': np.random.uniform(0.9, 1.0),
                'srv_diff_host_rate': np.random.uniform(0.9, 1.0),
                'dst_host_count': np.random.poisson(100),
                'dst_host_srv_count': np.random.poisson(100),
                'dst_host_same_srv_rate': np.random.uniform(0.0, 0.1),
                'dst_host_diff_srv_rate': np.random.uniform(0.9, 1.0),
                'dst_host_same_src_port_rate': np.random.uniform(0.0, 0.1),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.9, 1.0),
                'dst_host_serror_rate': np.random.uniform(0.8, 1.0),
                'dst_host_srv_serror_rate': np.random.uniform(0.8, 1.0),
                'dst_host_rerror_rate': np.random.uniform(0.0, 0.2),
                'dst_host_srv_rerror_rate': np.random.uniform(0.0, 0.2),
                'attack_type': 'ddos',
                'attack_category': 'dos'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_sql_injection_attack(self, num_samples: int = 300) -> pd.DataFrame:
        """Generate SQL injection attack data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(200),
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': np.random.poisson(500),
                'dst_bytes': np.random.poisson(2000),
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': np.random.poisson(3),
                'logged_in': 0,
                'num_compromised': np.random.poisson(1),
                'root_shell': np.random.choice([0, 1], p=[0.7, 0.3]),
                'su_attempted': 0,
                'num_root': np.random.choice([0, 1], p=[0.8, 0.2]),
                'num_file_creations': np.random.poisson(1),
                'num_shells': np.random.choice([0, 1], p=[0.9, 0.1]),
                'num_access_files': np.random.poisson(2),
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(10),
                'srv_count': np.random.poisson(8),
                'serror_rate': np.random.uniform(0.0, 0.3),
                'srv_serror_rate': np.random.uniform(0.0, 0.3),
                'rerror_rate': np.random.uniform(0.0, 0.2),
                'srv_rerror_rate': np.random.uniform(0.0, 0.2),
                'same_srv_rate': np.random.uniform(0.6, 0.9),
                'diff_srv_rate': np.random.uniform(0.1, 0.4),
                'srv_diff_host_rate': np.random.uniform(0.0, 0.2),
                'dst_host_count': np.random.poisson(15),
                'dst_host_srv_count': np.random.poisson(12),
                'dst_host_same_srv_rate': np.random.uniform(0.6, 0.9),
                'dst_host_diff_srv_rate': np.random.uniform(0.1, 0.4),
                'dst_host_same_src_port_rate': np.random.uniform(0.6, 0.9),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.0, 0.2),
                'dst_host_serror_rate': np.random.uniform(0.0, 0.3),
                'dst_host_srv_serror_rate': np.random.uniform(0.0, 0.3),
                'dst_host_rerror_rate': np.random.uniform(0.0, 0.2),
                'dst_host_srv_rerror_rate': np.random.uniform(0.0, 0.2),
                'attack_type': 'sql_injection',
                'attack_category': 'probe'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_xss_attack(self, num_samples: int = 200) -> pd.DataFrame:
        """Generate XSS attack data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(150),
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': np.random.poisson(800),
                'dst_bytes': np.random.poisson(1500),
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': np.random.poisson(1),
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(8),
                'srv_count': np.random.poisson(6),
                'serror_rate': np.random.uniform(0.0, 0.2),
                'srv_serror_rate': np.random.uniform(0.0, 0.2),
                'rerror_rate': np.random.uniform(0.0, 0.1),
                'srv_rerror_rate': np.random.uniform(0.0, 0.1),
                'same_srv_rate': np.random.uniform(0.7, 0.9),
                'diff_srv_rate': np.random.uniform(0.1, 0.3),
                'srv_diff_host_rate': np.random.uniform(0.0, 0.1),
                'dst_host_count': np.random.poisson(12),
                'dst_host_srv_count': np.random.poisson(10),
                'dst_host_same_srv_rate': np.random.uniform(0.7, 0.9),
                'dst_host_diff_srv_rate': np.random.uniform(0.1, 0.3),
                'dst_host_same_src_port_rate': np.random.uniform(0.7, 0.9),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.0, 0.1),
                'dst_host_serror_rate': np.random.uniform(0.0, 0.2),
                'dst_host_srv_serror_rate': np.random.uniform(0.0, 0.2),
                'dst_host_rerror_rate': np.random.uniform(0.0, 0.1),
                'dst_host_srv_rerror_rate': np.random.uniform(0.0, 0.1),
                'attack_type': 'xss',
                'attack_category': 'probe'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_brute_force_attack(self, num_samples: int = 250) -> pd.DataFrame:
        """Generate brute force attack data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(50),
                'protocol_type': 'tcp',
                'service': np.random.choice(['ssh', 'ftp', 'telnet']),
                'flag': np.random.choice(['SF', 'REJ']),
                'src_bytes': np.random.poisson(200),
                'dst_bytes': np.random.poisson(100),
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': np.random.poisson(10),
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(20),
                'srv_count': np.random.poisson(20),
                'serror_rate': np.random.uniform(0.0, 0.5),
                'srv_serror_rate': np.random.uniform(0.0, 0.5),
                'rerror_rate': np.random.uniform(0.0, 0.3),
                'srv_rerror_rate': np.random.uniform(0.0, 0.3),
                'same_srv_rate': np.random.uniform(0.5, 0.8),
                'diff_srv_rate': np.random.uniform(0.2, 0.5),
                'srv_diff_host_rate': np.random.uniform(0.0, 0.2),
                'dst_host_count': np.random.poisson(25),
                'dst_host_srv_count': np.random.poisson(25),
                'dst_host_same_srv_rate': np.random.uniform(0.5, 0.8),
                'dst_host_diff_srv_rate': np.random.uniform(0.2, 0.5),
                'dst_host_same_src_port_rate': np.random.uniform(0.5, 0.8),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.0, 0.2),
                'dst_host_serror_rate': np.random.uniform(0.0, 0.5),
                'dst_host_srv_serror_rate': np.random.uniform(0.0, 0.5),
                'dst_host_rerror_rate': np.random.uniform(0.0, 0.3),
                'dst_host_srv_rerror_rate': np.random.uniform(0.0, 0.3),
                'attack_type': 'brute_force',
                'attack_category': 'r2l'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_port_scan_attack(self, num_samples: int = 300) -> pd.DataFrame:
        """Generate port scanning attack data"""
        data = []
        
        for _ in range(num_samples):
            row = {
                'duration': np.random.exponential(30),
                'protocol_type': 'tcp',
                'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet']),
                'flag': np.random.choice(['S0', 'REJ', 'RSTO']),
                'src_bytes': np.random.poisson(100),
                'dst_bytes': 0,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': np.random.poisson(30),
                'srv_count': np.random.poisson(30),
                'serror_rate': np.random.uniform(0.0, 0.4),
                'srv_serror_rate': np.random.uniform(0.0, 0.4),
                'rerror_rate': np.random.uniform(0.0, 0.3),
                'srv_rerror_rate': np.random.uniform(0.0, 0.3),
                'same_srv_rate': np.random.uniform(0.0, 0.3),
                'diff_srv_rate': np.random.uniform(0.7, 1.0),
                'srv_diff_host_rate': np.random.uniform(0.7, 1.0),
                'dst_host_count': np.random.poisson(50),
                'dst_host_srv_count': np.random.poisson(50),
                'dst_host_same_srv_rate': np.random.uniform(0.0, 0.3),
                'dst_host_diff_srv_rate': np.random.uniform(0.7, 1.0),
                'dst_host_same_src_port_rate': np.random.uniform(0.0, 0.3),
                'dst_host_srv_diff_host_rate': np.random.uniform(0.7, 1.0),
                'dst_host_serror_rate': np.random.uniform(0.0, 0.4),
                'dst_host_srv_serror_rate': np.random.uniform(0.0, 0.4),
                'dst_host_rerror_rate': np.random.uniform(0.0, 0.3),
                'dst_host_srv_rerror_rate': np.random.uniform(0.0, 0.3),
                'attack_type': 'port_scan',
                'attack_category': 'probe'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_complete_dataset(self, total_samples: int = 5000) -> pd.DataFrame:
        """Generate a complete dataset with all attack types"""
        print("Generating cyber attack dataset...")
        
        # Calculate samples per category
        normal_samples = int(total_samples * 0.6)  # 60% normal traffic
        attack_samples = total_samples - normal_samples
        
        # Generate data for each category
        normal_data = self.generate_normal_traffic(normal_samples)
        ddos_data = self.generate_ddos_attack(int(attack_samples * 0.3))
        sql_data = self.generate_sql_injection_attack(int(attack_samples * 0.2))
        xss_data = self.generate_xss_attack(int(attack_samples * 0.15))
        brute_data = self.generate_brute_force_attack(int(attack_samples * 0.2))
        scan_data = self.generate_port_scan_attack(int(attack_samples * 0.15))
        
        # Combine all data
        combined_data = pd.concat([
            normal_data, ddos_data, sql_data, xss_data, brute_data, scan_data
        ], ignore_index=True)
        
        # Shuffle the data
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add timestamp
        combined_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            periods=len(combined_data),
            freq='S'
        )
        
        print(f"Generated dataset with {len(combined_data)} samples")
        print(f"Attack distribution:")
        print(combined_data['attack_type'].value_counts())
        
        return combined_data
    
    def save_dataset(self, data: pd.DataFrame, filename: str = 'cyber_attack_dataset.csv'):
        """Save the generated dataset to a file"""
        filepath = filename
        data.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath

if __name__ == "__main__":
    # Generate sample dataset
    generator = CyberAttackDataGenerator()
    dataset = generator.generate_complete_dataset(5000)
    generator.save_dataset(dataset)
