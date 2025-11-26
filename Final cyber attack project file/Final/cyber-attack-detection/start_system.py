#!/usr/bin/env python3
"""
Simple Startup Script for Cyber Attack Detection System
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    print("🛡️  CYBER ATTACK DETECTION SYSTEM")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('src'):
        print("❌ Error: Please run this script from the project root directory")
        return
    
    print("🚀 Starting the system...")
    
    # Start the API server
    print("\n📡 Starting API server...")
    try:
        api_process = subprocess.Popen([
            sys.executable, "src/api/app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        print("✅ API server started on http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return
    
    # Train models if needed
    print("\n🤖 Training models...")
    try:
        import requests
        response = requests.post('http://localhost:5000/train', json={})
        if response.status_code == 200:
            print("✅ Models trained successfully!")
        else:
            print("⚠️  Model training may have failed")
    except:
        print("⚠️  Could not train models (server may still be starting)")
    
    # Open the web interface
    print("\n🌐 Opening web interface...")
    try:
        # Get the absolute path to the HTML file
        html_path = Path(__file__).parent / "index.html"
        file_url = f'file:///{html_path.absolute().as_posix()}'
        print(f"📂 Opening: {file_url}")
        webbrowser.open(file_url)
        print("✅ Home page opened in your browser")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually open 'index.html' in your browser")
        print("   Or navigate to: file:///" + str(Path(__file__).parent.absolute() / "index.html").replace("\\", "/"))
    
    print("\n" + "=" * 50)
    print("🎉 SYSTEM IS READY!")
    print("=" * 50)
    print("✅ API Server: http://localhost:5000")
    print("✅ Home Page: index.html")
    print("✅ Demo Interface: upload_demo.html")
    print("✅ Sample Files:")
    print("   - sample_testing.csv (all attack types)")
    print("   - sample_data_template.csv (normal traffic)")
    print("   - sample_ddos_attack.csv (DDoS attacks)")
    print("\n📝 How to use:")
    print("1. Explore the home page to learn about the system")
    print("2. Click 'Try Demo Now' to test the detection system")
    print("3. Upload any CSV file and analyze for cyber attacks")
    print("\n🛑 Press Ctrl+C to stop the system")
    
    try:
        # Keep the script running
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping the system...")
        api_process.terminate()
        print("✅ System stopped")

if __name__ == "__main__":
    main()
