"""
Startup script for Quantum-Enhanced AI Reasoning Architecture.

This script launches all components needed for the quantum reasoning system:
- Quantum Attention API server
- Truth Verification API server  
- System health checks
- Integration testing
"""

import subprocess
import time
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumReasoningLauncher:
    """Launcher for quantum reasoning architecture components."""
    
    def __init__(self):
        """Initialize launcher."""
        self.base_path = Path(__file__).parent
        self.quantum_attention_port = 8001
        self.truth_verification_port = 8002
        self.processes = []
        
        # Component status
        self.component_status = {
            'quantum_attention_api': False,
            'truth_verification_api': False,
            'ollama_service': False
        }
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama service is running.
        
        Returns:
            True if Ollama is running
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"Ollama is running with {len(models)} models available")
                for model in models[:3]:  # Show first 3 models
                    logger.info(f"  - {model.get('name', 'unknown')}")
                return True
        except Exception as e:
            logger.warning(f"Ollama not accessible: {e}")
        
        return False
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available
        """
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_quantum_attention_api(self) -> bool:
        """Start the Quantum Attention API server.
        
        Returns:
            True if started successfully
        """
        logger.info("Starting Quantum Attention API...")
        
        api_script = self.base_path / "quantum_attention_api.py"
        if not api_script.exists():
            logger.error(f"Quantum Attention API script not found: {api_script}")
            return False
        
        try:
            # Start the API server
            process = subprocess.Popen([
                sys.executable, 
                str(api_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(('quantum_attention_api', process))
            
            # Wait for startup
            for attempt in range(10):
                time.sleep(2)
                if self.check_port_available(self.quantum_attention_port):
                    logger.info(f"Quantum Attention API started on port {self.quantum_attention_port}")
                    self.component_status['quantum_attention_api'] = True
                    return True
            
            logger.error("Quantum Attention API failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Quantum Attention API: {e}")
            return False
    
    def start_truth_verification_api(self) -> bool:
        """Start the Truth Verification API server.
        
        Returns:
            True if started successfully
        """
        logger.info("Starting Truth Verification API...")
        
        api_script = self.base_path / "truth_verification_api.py"
        if not api_script.exists():
            logger.error(f"Truth Verification API script not found: {api_script}")
            return False
        
        try:
            # Start the API server
            process = subprocess.Popen([
                sys.executable,
                str(api_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(('truth_verification_api', process))
            
            # Wait for startup
            for attempt in range(10):
                time.sleep(2)
                if self.check_port_available(self.truth_verification_port):
                    logger.info(f"Truth Verification API started on port {self.truth_verification_port}")
                    self.component_status['truth_verification_api'] = True
                    return True
            
            logger.error("Truth Verification API failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Truth Verification API: {e}")
            return False
    
    def test_quantum_attention(self) -> bool:
        """Test quantum attention functionality.
        
        Returns:
            True if test passed
        """
        logger.info("Testing Quantum Attention API...")
        
        try:
            test_payload = {
                "query": "Is it ethical to use AI for hiring decisions?",
                "context": "Consider both fairness and efficiency perspectives",
                "force_collapse": False,
                "num_heads": 4,
                "num_interpretations": 3
            }
            
            response = requests.post(
                f"http://localhost:{self.quantum_attention_port}/quantum-attention",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Quantum Attention test PASSED")
                logger.info(f"  Processing time: {result.get('processing_time', 0):.2f}s")
                logger.info(f"  Success: {result.get('success', False)}")
                return True
            else:
                logger.error(f"Quantum Attention test FAILED: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Quantum Attention test ERROR: {e}")
            return False
    
    def test_truth_verification(self) -> bool:
        """Test truth verification functionality.
        
        Returns:
            True if test passed
        """
        logger.info("Testing Truth Verification API...")
        
        try:
            test_payload = {
                "query": "What is the capital of France and why is it economically important?",
                "context": "Consider both historical and modern perspectives",
                "num_agents": 3,
                "consistency_threshold": 0.7
            }
            
            response = requests.post(
                f"http://localhost:{self.truth_verification_port}/truth-verification",
                json=test_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Truth Verification test PASSED")
                logger.info(f"  Processing time: {result.get('processing_time', 0):.2f}s")
                logger.info(f"  Verification result: {result.get('verification_result', 'unknown')}")
                logger.info(f"  Consistency score: {result.get('consistency_score', 0):.2f}")
                return True
            else:
                logger.error(f"Truth Verification test FAILED: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Truth Verification test ERROR: {e}")
            return False
    
    def display_integration_info(self):
        """Display information about N8n integration."""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION INFORMATION")
        logger.info("="*60)
        
        logger.info("\n📋 N8n Workflow Integration:")
        logger.info("1. Import the enhanced JARVIS workflow:")
        logger.info(f"   File: {self.base_path / 'jarvis_enhanced_workflow.json'}")
        
        logger.info("\n🔗 API Endpoints Available:")
        logger.info(f"   Quantum Attention: http://localhost:{self.quantum_attention_port}")
        logger.info(f"   Truth Verification: http://localhost:{self.truth_verification_port}")
        
        logger.info("\n📱 Telegram Integration:")
        logger.info("   Configure your Telegram bot webhook to point to N8n")
        logger.info("   Enhanced JARVIS will automatically route queries to quantum tools")
        
        logger.info("\n🧠 Enhanced Capabilities:")
        logger.info("   - Quantum superposition attention for complex reasoning")
        logger.info("   - Multi-agent truth verification for critical decisions")
        logger.info("   - Quantum entanglement between reasoning chains")
        logger.info("   - Automatic routing based on query complexity")
        
        logger.info("\n🎯 Example Queries:")
        logger.info("   High-stakes: 'Should I invest my retirement in crypto?'")
        logger.info("   Complex reasoning: 'Is AI consciousness possible?'")
        logger.info("   Ethical dilemma: 'Should we use AI for hiring?'")
        
        logger.info("\n" + "="*60)
    
    def display_status_dashboard(self):
        """Display current system status."""
        logger.info("\n" + "🚀 QUANTUM REASONING ARCHITECTURE STATUS")
        logger.info("="*60)
        
        # Component status
        for component, status in self.component_status.items():
            status_icon = "✅" if status else "❌"
            component_name = component.replace('_', ' ').title()
            logger.info(f"{status_icon} {component_name}")
        
        # Ollama status
        ollama_status = self.check_ollama_status()
        ollama_icon = "✅" if ollama_status else "❌"
        logger.info(f"{ollama_icon} Ollama Service")
        
        logger.info("\n📊 System Metrics:")
        logger.info(f"   Active Processes: {len(self.processes)}")
        logger.info(f"   Quantum Attention Port: {self.quantum_attention_port}")
        logger.info(f"   Truth Verification Port: {self.truth_verification_port}")
        logger.info(f"   Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall status
        all_critical_running = (
            self.component_status['quantum_attention_api'] and 
            self.component_status['truth_verification_api']
        )
        
        if all_critical_running:
            logger.info("\n🎉 ALL SYSTEMS OPERATIONAL!")
            logger.info("   Ready for quantum-enhanced reasoning")
        else:
            logger.info("\n⚠️  SOME SYSTEMS NOT OPERATIONAL")
            logger.info("   Check individual component status above")
    
    def run_comprehensive_test(self):
        """Run comprehensive system tests."""
        logger.info("\n🧪 RUNNING COMPREHENSIVE TESTS")
        logger.info("="*50)
        
        tests = [
            ("Quantum Attention", self.test_quantum_attention),
            ("Truth Verification", self.test_truth_verification)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {e}")
        
        logger.info(f"\n📈 TEST RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - System ready for deployment!")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed - Check logs above")
    
    def start_all_services(self):
        """Start all quantum reasoning services."""
        logger.info("🚀 STARTING QUANTUM-ENHANCED AI REASONING ARCHITECTURE")
        logger.info("="*70)
        
        # Check Ollama first
        self.component_status['ollama_service'] = self.check_ollama_status()
        if not self.component_status['ollama_service']:
            logger.warning("⚠️  Ollama not running - Truth verification will use mock mode")
            logger.info("   Start Ollama with: ollama serve")
        
        # Start APIs
        success_count = 0
        
        if self.start_quantum_attention_api():
            success_count += 1
        
        if self.start_truth_verification_api():
            success_count += 1
        
        # Display results
        self.display_status_dashboard()
        
        if success_count >= 1:
            time.sleep(2)  # Let services fully initialize
            self.run_comprehensive_test()
            self.display_integration_info()
            
            logger.info("\n💡 Next Steps:")
            logger.info("1. Import N8n workflow from jarvis_enhanced_workflow.json")
            logger.info("2. Configure Telegram bot webhook")
            logger.info("3. Test with enhanced JARVIS queries")
            logger.info("4. Monitor system performance")
            
            # Keep services running
            logger.info("\n🔄 Services running... Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(30)
                    # Periodic health check
                    for component in ['quantum_attention_api', 'truth_verification_api']:
                        if self.component_status[component]:
                            port = self.quantum_attention_port if 'attention' in component else self.truth_verification_port
                            if not self.check_port_available(port):
                                logger.warning(f"⚠️  {component} appears to have stopped")
                                self.component_status[component] = False
            
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutdown requested...")
                self.stop_all_services()
        
        else:
            logger.error("❌ Failed to start critical services")
            self.stop_all_services()
    
    def stop_all_services(self):
        """Stop all running services."""
        logger.info("🛑 Stopping all services...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.info(f"🔪 Force killed {name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        logger.info("🏁 All services stopped")


def main():
    """Main entry point."""
    launcher = QuantumReasoningLauncher()
    
    try:
        launcher.start_all_services()
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        launcher.stop_all_services()


if __name__ == "__main__":
    main()
