"""
Hopsworks Connection Test
==========================
Purpose: Initial connection test to verify Hopsworks setup
Status: Reference only - not used in production pipeline
Date: Project setup phase

This was used to verify:
- API key configuration
- Project access
- Feature Store availability
- Model Registry availability
"""

import hopsworks
import os
from dotenv import load_dotenv

class HopsworksIntegration:
    """Simple connection test wrapper for Hopsworks"""
    
    def __init__(self):
        load_dotenv()
        
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
        
        self.project = None
        self.fs = None
        self.mr = None
        
        print("🔍 ENV CHECK")
        print("API KEY loaded:", "YES" if self.api_key else "NO")
        print("Project name:", self.project_name)
    
    def connect(self):
        """Connect to Hopsworks and verify access"""
        try:
            print("🔗 Connecting to Hopsworks...")
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            print(f"✅ Connected to project: {self.project_name}")
            return True
        except Exception as e:
            print(f"❌ Hopsworks connection failed: {e}")
            return False


# Quick test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 HOPSWORKS CONNECTION TEST")
    print("=" * 60)
    
    hw = HopsworksIntegration()
    
    if hw.connect():
        print("\n✅ Feature Store ready")
        print("✅ Model Registry ready")
        print("\n" + "=" * 60)
        print("🎉 HOPSWORKS IS READY!")
        print("=" * 60)
    else:
        print("\n❌ Connection failed - check credentials")