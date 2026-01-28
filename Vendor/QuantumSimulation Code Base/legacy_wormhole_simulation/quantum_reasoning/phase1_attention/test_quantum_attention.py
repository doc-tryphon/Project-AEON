"""
Testing framework for Quantum Superposition Attention.

This module provides comprehensive tests for quantum attention mechanisms,
including contradictory input handling, superposition maintenance, and collapse behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

from quantum_attention import MultiHeadQuantumAttention, QuantumAttentionHead


class QuantumAttentionTester:
    """Comprehensive testing suite for quantum attention mechanisms."""
    
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        """Initialize tester with quantum attention model.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.model = MultiHeadQuantumAttention(d_model=d_model, num_heads=num_heads)
        self.test_results = {}
    
    def generate_contradictory_inputs(self, batch_size: int = 2, seq_len: int = 10) -> Dict[str, torch.Tensor]:
        """Generate inputs that should trigger contradictory interpretations.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary of test inputs
        """
        # Create base inputs
        query = torch.randn(batch_size, seq_len, self.d_model)
        
        # Create contradictory key-value pairs
        # First half of sequence suggests one interpretation
        # Second half suggests opposite interpretation
        key = torch.randn(batch_size, seq_len, self.d_model)
        value = torch.randn(batch_size, seq_len, self.d_model)
        
        # Inject contradictory patterns
        for i in range(batch_size):
            # Make first half correlate strongly with query
            key[i, :seq_len//2, :] = query[i, :seq_len//2, :] + 0.1 * torch.randn(seq_len//2, self.d_model)
            
            # Make second half anti-correlate with query
            key[i, seq_len//2:, :] = -query[i, seq_len//2:, :] + 0.1 * torch.randn(seq_len//2, self.d_model)
            
            # Create opposing value patterns
            value[i, :seq_len//2, :] = torch.ones(seq_len//2, self.d_model)
            value[i, seq_len//2:, :] = -torch.ones(seq_len//2, self.d_model)
        
        return {
            'query': query,
            'key': key,
            'value': value,
            'description': 'Contradictory key-value patterns'
        }
    
    def test_superposition_maintenance(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Test ability to maintain superposition with contradictory inputs.
        
        Args:
            test_data: Test input tensors
            
        Returns:
            Test results
        """
        print("Testing superposition maintenance...")
        
        output, metadata = self.model(
            test_data['query'],
            test_data['key'],
            test_data['value'],
            force_collapse=False
        )
        
        results = {
            'test_name': 'superposition_maintenance',
            'heads_in_superposition': metadata['heads_in_superposition'],
            'total_heads': metadata['total_heads'],
            'superposition_ratio': metadata['heads_in_superposition'] / metadata['total_heads'],
            'average_entropy': metadata['average_entropy'],
            'output_variance': torch.var(output).item(),
            'metadata': metadata
        }
        
        # Success criterion: at least 50% of heads maintain superposition
        results['success'] = results['superposition_ratio'] >= 0.5
        
        print(f"Superposition ratio: {results['superposition_ratio']:.2f}")
        print(f"Average entropy: {results['average_entropy']:.4f}")
        print(f"Test {'PASSED' if results['success'] else 'FAILED'}")
        
        return results
    
    def test_forced_collapse(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Test forced collapse behavior.
        
        Args:
            test_data: Test input tensors
            
        Returns:
            Test results
        """
        print("Testing forced collapse...")
        
        output, metadata = self.model(
            test_data['query'],
            test_data['key'],
            test_data['value'],
            force_collapse=True
        )
        
        results = {
            'test_name': 'forced_collapse',
            'heads_in_superposition': metadata['heads_in_superposition'],
            'total_heads': metadata['total_heads'],
            'superposition_ratio': metadata['heads_in_superposition'] / metadata['total_heads'],
            'average_entropy': metadata['average_entropy'],
            'output_variance': torch.var(output).item(),
            'metadata': metadata
        }
        
        # Success criterion: no heads should maintain superposition
        results['success'] = results['superposition_ratio'] == 0
        
        print(f"Superposition ratio: {results['superposition_ratio']:.2f}")
        print(f"Average entropy: {results['average_entropy']:.4f}")
        print(f"Test {'PASSED' if results['success'] else 'FAILED'}")
        
        return results
    
    def test_consistency_detection(self, num_trials: int = 100) -> Dict[str, Any]:
        """Test ability to detect and handle inconsistent interpretations.
        
        Args:
            num_trials: Number of test trials
            
        Returns:
            Test results
        """
        print("Testing consistency detection...")
        
        collapse_triggers = 0
        superposition_maintained = 0
        entropy_values = []
        
        for trial in range(num_trials):
            # Generate random contradictory inputs
            test_data = self.generate_contradictory_inputs()
            
            output, metadata = self.model(
                test_data['query'],
                test_data['key'],
                test_data['value'],
                force_collapse=False
            )
            
            if metadata['heads_in_superposition'] < metadata['total_heads']:
                collapse_triggers += 1
            else:
                superposition_maintained += 1
            
            entropy_values.append(metadata['average_entropy'])
        
        results = {
            'test_name': 'consistency_detection',
            'trials': num_trials,
            'collapse_triggers': collapse_triggers,
            'superposition_maintained': superposition_maintained,
            'collapse_rate': collapse_triggers / num_trials,
            'average_entropy': np.mean(entropy_values),
            'entropy_std': np.std(entropy_values),
            'entropy_values': entropy_values
        }
        
        # Success criterion: system should collapse some contradictory cases
        results['success'] = 0.1 <= results['collapse_rate'] <= 0.9
        
        print(f"Collapse rate: {results['collapse_rate']:.2f}")
        print(f"Average entropy: {results['average_entropy']:.4f} ± {results['entropy_std']:.4f}")
        print(f"Test {'PASSED' if results['success'] else 'FAILED'}")
        
        return results
    
    def test_interpretation_diversity(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Test diversity of simultaneous interpretations.
        
        Args:
            test_data: Test input tensors
            
        Returns:
            Test results
        """
        print("Testing interpretation diversity...")
        
        # Run multiple trials to gather interpretation statistics
        interpretation_counts = {}
        entropy_values = []
        
        for trial in range(50):
            output, metadata = self.model(
                test_data['query'],
                test_data['key'],
                test_data['value'],
                force_collapse=False
            )
            
            entropy_values.append(metadata['average_entropy'])
            
            # Analyze head metadata for interpretation diversity
            for head_meta in metadata['head_metadata']:
                if head_meta.get('superposition_maintained', False):
                    num_interpretations = head_meta.get('num_active_interpretations', 0)
                    interpretation_counts[num_interpretations] = interpretation_counts.get(num_interpretations, 0) + 1
        
        results = {
            'test_name': 'interpretation_diversity',
            'interpretation_distribution': interpretation_counts,
            'average_entropy': np.mean(entropy_values),
            'entropy_std': np.std(entropy_values),
            'max_interpretations': max(interpretation_counts.keys()) if interpretation_counts else 0,
            'trials': 50
        }
        
        # Success criterion: should maintain multiple interpretations with high entropy
        results['success'] = results['average_entropy'] > 0.5 and results['max_interpretations'] >= 2
        
        print(f"Interpretation distribution: {interpretation_counts}")
        print(f"Average entropy: {results['average_entropy']:.4f} ± {results['entropy_std']:.4f}")
        print(f"Max interpretations: {results['max_interpretations']}")
        print(f"Test {'PASSED' if results['success'] else 'FAILED'}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report.
        
        Returns:
            Complete test results
        """
        print("=" * 60)
        print("QUANTUM SUPERPOSITION ATTENTION - COMPREHENSIVE TEST")
        print("=" * 60)
        
        # Generate test data
        test_data = self.generate_contradictory_inputs()
        
        # Run all tests
        tests = [
            self.test_superposition_maintenance(test_data),
            self.test_forced_collapse(test_data),
            self.test_consistency_detection(),
            self.test_interpretation_diversity(test_data)
        ]
        
        # Aggregate results
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests if test['success'])
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'd_model': self.d_model,
                'num_heads': self.num_heads
            },
            'individual_tests': tests,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': passed_tests / total_tests,
                'overall_success': passed_tests == total_tests
            }
        }
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Pass rate: {comprehensive_results['summary']['pass_rate']:.2%}")
        print(f"Overall result: {'SUCCESS' if comprehensive_results['summary']['overall_success'] else 'FAILURE'}")
        
        self.test_results = comprehensive_results
        return comprehensive_results
    
    def save_results(self, filename: str = None):
        """Save test results to file.
        
        Args:
            filename: Output filename (optional)
        """
        if not self.test_results:
            print("No test results to save. Run tests first.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_attention_test_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            return obj
        
        serializable_results = json.loads(json.dumps(self.test_results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Test results saved to: {filename}")
    
    def visualize_results(self):
        """Create visualizations of test results."""
        if not self.test_results:
            print("No test results to visualize. Run tests first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quantum Attention Test Results', fontsize=16)
        
        # Test pass rates
        ax1 = axes[0, 0]
        test_names = [test['test_name'].replace('_', ' ').title() for test in self.test_results['individual_tests']]
        pass_status = [test['success'] for test in self.test_results['individual_tests']]
        colors = ['green' if passed else 'red' for passed in pass_status]
        ax1.bar(range(len(test_names)), [1 if p else 0 for p in pass_status], color=colors)
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.set_ylabel('Pass (1) / Fail (0)')
        ax1.set_title('Test Pass/Fail Status')
        ax1.set_ylim(0, 1.2)
        
        # Superposition ratios
        ax2 = axes[0, 1]
        superposition_tests = [test for test in self.test_results['individual_tests'] 
                              if 'superposition_ratio' in test]
        if superposition_tests:
            names = [test['test_name'].replace('_', ' ').title() for test in superposition_tests]
            ratios = [test['superposition_ratio'] for test in superposition_tests]
            ax2.bar(range(len(names)), ratios, color='blue', alpha=0.7)
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.set_ylabel('Superposition Ratio')
            ax2.set_title('Heads Maintaining Superposition')
            ax2.set_ylim(0, 1.2)
        
        # Entropy distribution
        ax3 = axes[1, 0]
        consistency_test = next((test for test in self.test_results['individual_tests'] 
                               if test['test_name'] == 'consistency_detection'), None)
        if consistency_test and 'entropy_values' in consistency_test:
            ax3.hist(consistency_test['entropy_values'], bins=20, alpha=0.7, color='purple')
            ax3.axvline(consistency_test['average_entropy'], color='red', linestyle='--', 
                       label=f"Mean: {consistency_test['average_entropy']:.3f}")
            ax3.set_xlabel('Quantum Entropy')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Entropy Distribution')
            ax3.legend()
        
        # Interpretation diversity
        ax4 = axes[1, 1]
        diversity_test = next((test for test in self.test_results['individual_tests'] 
                              if test['test_name'] == 'interpretation_diversity'), None)
        if diversity_test and 'interpretation_distribution' in diversity_test:
            dist = diversity_test['interpretation_distribution']
            interpretations = list(dist.keys())
            counts = list(dist.values())
            ax4.bar(interpretations, counts, color='orange', alpha=0.7)
            ax4.set_xlabel('Number of Active Interpretations')
            ax4.set_ylabel('Count')
            ax4.set_title('Interpretation Diversity Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'quantum_attention_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: quantum_attention_results_{timestamp}.png")


def main():
    """Main testing function."""
    print("Initializing Quantum Attention Tester...")
    
    # Initialize tester
    tester = QuantumAttentionTester(d_model=256, num_heads=8)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Save results
    tester.save_results()
    
    # Create visualizations
    try:
        tester.visualize_results()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Results saved to file successfully.")
    
    return results


if __name__ == "__main__":
    main()
