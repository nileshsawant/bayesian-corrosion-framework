#!/projects/hpcapps/nsawant/corrosion/env/bin/python3
"""
Active Learning Pipeline for BNN Corrosion Model
================================================
Intelligently switches between fast BNN predictions and expensive physics 
simulations based on prediction uncertainty.

Usage:
    python3 active_learning.py --model bnn_model.pt --data training_data.pkl --params new_conditions.txt
    
    # Single prediction with active learning:
    python3 active_learning.py --model bnn_model.pt --nacl 0.3 --temp 290 --ph 7.0 --flow 2.5
    
    # Batch mode with auto-retraining:
    python3 active_learning.py --model bnn_model.pt --params batch.txt --retrain-every 10
"""

import argparse
import sys
import os
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bnn_model import BNNWrapper
from physics_bridge import PhysicsBridge
from train_bnn import train_bnn_batch


class ActiveLearningPipeline:
    """
    Active learning pipeline that uses BNN for confident predictions
    and falls back to physics simulations for uncertain cases.
    """
    
    def __init__(self, 
                 model_path,
                 dataset_path=None,
                 uncertainty_threshold=0.05,
                 retrain_every=1,
                 dependency_root='../corrosion-modeling-applications',
                 output_dir='active_learning_results'):
        """
        Args:
            model_path: Path to trained BNN model
            dataset_path: Path to training dataset (for updates)
            uncertainty_threshold: Relative uncertainty threshold (default: 5%)
            retrain_every: Retrain after accumulating this many new samples (default: 1)
            dependency_root: Path to physics simulation code
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.uncertainty_threshold = uncertainty_threshold
        self.retrain_every = retrain_every
        
        # Load BNN model
        print(f"Loading BNN model from: {model_path}")
        self.bnn = BNNWrapper.load(model_path)
        
        # Load existing dataset if provided
        self.dataset = None
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"Loaded dataset: {len(self.dataset['inputs'])} samples")
        
        # Initialize physics bridge
        self.physics = PhysicsBridge(dependency_root)
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Track new samples and statistics
        self.new_samples = []
        self.stats = {
            'total_predictions': 0,
            'bnn_used': 0,
            'physics_used': 0,
            'retrains': 0,
            'avg_uncertainty': []
        }
    
    def predict_with_uncertainty(self, params, num_samples=100):
        """
        Run BNN prediction and calculate uncertainty metrics.
        
        Args:
            params: dict with keys ['NaCl', 'Temp', 'pH', 'Flow']
            num_samples: Number of posterior samples for uncertainty
            
        Returns:
            pred_mean, pred_std, relative_uncertainty
        """
        param_array = np.array([[
            params['NaCl'],
            params['Temp'],
            params['pH'],
            params['Flow']
        ]])
        
        X = torch.tensor(param_array).float()
        
        with torch.no_grad():
            pred_mean, pred_std = self.bnn.predict(X, num_samples=num_samples)
        
        pred_mean = pred_mean[0].cpu().numpy()
        pred_std = pred_std[0].cpu().numpy()
        
        # Calculate relative uncertainty (std / |mean|)
        # Focus on phi portion (first 2541 values)
        phi_len = 121 * 21
        phi_mean = pred_mean[:phi_len]
        phi_std = pred_std[:phi_len]
        
        rel_uncertainty = np.mean(phi_std) / np.abs(phi_mean).mean()
        
        return pred_mean, pred_std, rel_uncertainty
    
    def run_physics_simulation(self, params):
        """
        Run expensive physics simulation.
        
        Args:
            params: dict with keys ['NaCl', 'Temp', 'pH', 'Flow']
            
        Returns:
            output_vector, metadata
        """
        print(f"  → Running physics simulation...")
        param_list = [params['NaCl'], params['Temp'], params['pH'], params['Flow']]
        
        result = self.physics.run_sim(param_list)
        
        if result is None:
            raise RuntimeError("Physics simulation failed")
        
        # Parse result
        phi_field = result['phi']
        j_profile = result['currentDensity']
        corr_rate = result.get('scalar', result.get('corrosionRate', 0.0))  # Handle both key names
        phi_shape = phi_field.shape  # Get shape directly from numpy array
        
        # Create output vector [phi, J]
        output = np.concatenate([phi_field.flatten(), j_profile])
        
        metadata = {
            'params': params,
            'phi_shape': phi_shape,
            'phi_length': len(phi_field.flatten()),
            'current_density_length': len(j_profile),
            'output_length': len(output),
            'corrosion_rate': corr_rate,
            'simulation_time': datetime.now().isoformat()
        }
        
        return output, metadata
    
    def predict(self, params, force_physics=False):
        """
        Main prediction function with active learning.
        
        Args:
            params: dict with keys ['NaCl', 'Temp', 'pH', 'Flow']
            force_physics: If True, skip BNN and use physics
            
        Returns:
            result: dict with prediction, uncertainty, and source
        """
        self.stats['total_predictions'] += 1
        
        print(f"\n{'='*70}")
        print(f"Prediction #{self.stats['total_predictions']}")
        print(f"Parameters: NaCl={params['NaCl']:.3f}M, T={params['Temp']:.1f}K, "
              f"pH={params['pH']:.2f}, Flow={params['Flow']:.3f}m/s")
        print(f"{'='*70}")
        
        if force_physics:
            print("⚠ Physics simulation forced")
            use_physics = True
            rel_uncertainty = None
            pred_mean = None
            pred_std = None
        else:
            # Try BNN first
            print("→ Running BNN inference...")
            pred_mean, pred_std, rel_uncertainty = self.predict_with_uncertainty(params)
            
            self.stats['avg_uncertainty'].append(rel_uncertainty)
            
            print(f"  Relative uncertainty: {rel_uncertainty*100:.2f}%")
            print(f"  Threshold: {self.uncertainty_threshold*100:.2f}%")
            
            use_physics = rel_uncertainty > self.uncertainty_threshold
        
        if use_physics:
            # High uncertainty or forced - use physics
            print(f"{'⚠ HIGH UNCERTAINTY' if not force_physics else ''} - Using physics simulation")
            self.stats['physics_used'] += 1
            
            output, metadata = self.run_physics_simulation(params)
            
            # Add to new samples for retraining
            param_array = np.array([params['NaCl'], params['Temp'], params['pH'], params['Flow']])
            self.new_samples.append({
                'input': param_array,
                'output': output,
                'metadata': metadata
            })
            
            print(f"  ✓ Physics simulation complete")
            print(f"  → New samples collected: {len(self.new_samples)}")
            
            # Check if we should retrain
            if len(self.new_samples) >= self.retrain_every:
                self.retrain_model()
            
            result = {
                'prediction': output,
                'uncertainty': None,
                'source': 'physics',
                'params': params,
                'corrosion_rate': metadata['corrosion_rate']
            }
        else:
            # Low uncertainty - trust BNN
            print(f"✓ LOW UNCERTAINTY - Using BNN prediction")
            self.stats['bnn_used'] += 1
            
            # Extract J profile and calculate corrosion rate
            phi_len = 121 * 21
            j_len = 60
            j_pred = pred_mean[phi_len:phi_len+j_len]
            corr_rate = np.mean(j_pred)
            
            result = {
                'prediction': pred_mean,
                'uncertainty': pred_std,
                'relative_uncertainty': rel_uncertainty,
                'source': 'bnn',
                'params': params,
                'corrosion_rate': corr_rate
            }
        
        self.print_result(result)
        try:
            self.save_and_plot_result(result)
        except Exception as e:
            print(f"⚠ Warning: Failed to save results: {e}")
            import traceback
            traceback.print_exc()
        return result
    
    def print_result(self, result):
        """Pretty print prediction result"""
        print(f"\n{'-'*70}")
        print(f"RESULT:")
        print(f"  Source: {result['source'].upper()}")
        print(f"  Corrosion Rate: {result['corrosion_rate']:.4e} A/m²")
        
        if result['source'] == 'bnn':
            print(f"  Confidence: {(1-result['relative_uncertainty'])*100:.1f}%")
        
        print(f"{'-'*70}")
    
    def save_and_plot_result(self, result):
        """Save result data and generate plots"""
        # Create filename based on parameters and timestamp
        params = result['params']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"pred_{self.stats['total_predictions']:03d}_{timestamp}"
        prefix += f"_NaCl{params['NaCl']:.2f}_T{params['Temp']:.0f}_pH{params['pH']:.1f}_Flow{params['Flow']:.2f}"
        
        # Extract phi and J profiles
        phi_len = 121 * 21  # 121 x-points, 21 y-points
        j_len = 60  # 60 boundary points
        
        output = result['prediction']
        phi_array = output[:phi_len].reshape(121, 21)
        j_array = output[phi_len:phi_len+j_len]
        
        # Save raw data (pickle)
        data_file = self.output_dir / f"{prefix}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'params': params,
                'phi': phi_array,
                'current_density': j_array,
                'corrosion_rate': result['corrosion_rate'],
                'source': result['source'],
                'uncertainty': result.get('uncertainty'),
                'relative_uncertainty': result.get('relative_uncertainty'),
                'timestamp': timestamp
            }, f)
        
        # Save summary (CSV)
        summary_file = self.output_dir / f"{prefix}_summary.csv"
        with open(summary_file, 'w') as f:
            f.write("Parameter,Value\n")
            f.write(f"NaCl_M,{params['NaCl']}\n")
            f.write(f"Temperature_K,{params['Temp']}\n")
            f.write(f"pH,{params['pH']}\n")
            f.write(f"Flow_m_per_s,{params['Flow']}\n")
            f.write(f"Corrosion_Rate_A_per_m2,{result['corrosion_rate']}\n")
            f.write(f"Source,{result['source']}\n")
            if result['source'] == 'bnn':
                f.write(f"Relative_Uncertainty,{result['relative_uncertainty']}\n")
                f.write(f"Confidence_Percent,{(1-result['relative_uncertainty'])*100:.2f}\n")
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Prediction #{self.stats['total_predictions']} - {result['source'].upper()}\n"
                     f"NaCl={params['NaCl']:.3f}M, T={params['Temp']:.1f}K, pH={params['pH']:.2f}, Flow={params['Flow']:.3f}m/s",
                     fontsize=12, fontweight='bold')
        
        # Define geometry (from run_physics.m)
        # Total domain: 6m (3m anode + 3m cathode), height: 1m
        x = np.linspace(0, 6.0, 121)  # 0 to 6m (121 nodes, 0.05m spacing)
        y = np.linspace(0, 1.0, 21)   # 0 to 1m (21 nodes, 0.05m spacing)
        X, Y = np.meshgrid(x, y)
        
        # Plot 1: Potential contour
        ax1 = axes[0, 0]
        contour = ax1.contourf(X, Y, phi_array.T, levels=20, cmap='RdBu_r')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Electric Potential φ(x,y)\n(Model convention: cathode=negative, anode=positive)')
        plt.colorbar(contour, ax=ax1, label='φ (V)')
        ax1.set_aspect('equal')
        # Mark materials
        ax1.axvline(x=3.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(1.5, 0.9, 'I625\n(Cathode)', ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax1.text(4.5, 0.9, 'CuNi\n(Anode)', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Plot 2: Potential along anode boundary (y=0)
        ax2 = axes[0, 1]
        phi_anode = phi_array[:, 0]  # First row (y=0)
        ax2.plot(x, phi_anode, 'b-', linewidth=2, label='Anode (y=0)')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('Potential φ (V)')
        ax2.set_title('Potential along Anode Boundary')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Current density distribution
        ax3 = axes[1, 0]
        # Anode is nodes 61-120 (x = 3m to 6m), cathode is nodes 1-60 (x = 0 to 3m)
        x_anode = x[60:120]  # Last 60 points (anode portion from 3m to 6m)
        ax3.plot(x_anode, j_array, 'r-', linewidth=2, label='Current Density')
        ax3.axhline(y=result['corrosion_rate'], color='g', linestyle='--', 
                    linewidth=1.5, label=f'Average: {result["corrosion_rate"]:.3e} A/m²')
        ax3.axvline(x=3.0, color='k', linestyle=':', linewidth=1, label='Cathode-Anode Junction')
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('Current Density J (A/m²)')
        ax3.set_title('Current Density Distribution (Anode)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Statistics and info
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = [
            "PREDICTION SUMMARY",
            "=" * 40,
            f"Source: {result['source'].upper()}",
            f"Corrosion Rate: {result['corrosion_rate']:.4e} A/m²",
            "",
        ]
        
        if result['source'] == 'bnn':
            info_text.extend([
                f"Relative Uncertainty: {result['relative_uncertainty']*100:.2f}%",
                f"Confidence: {(1-result['relative_uncertainty'])*100:.1f}%",
                "",
            ])
        
        info_text.extend([
            "PARAMETERS",
            "-" * 40,
            f"NaCl Concentration: {params['NaCl']:.3f} M",
            f"Temperature: {params['Temp']:.1f} K",
            f"pH: {params['pH']:.2f}",
            f"Flow Velocity: {params['Flow']:.3f} m/s",
            "",
            "OUTPUT STATISTICS",
            "-" * 40,
            f"φ min: {phi_array.min():.4f} V",
            f"φ max: {phi_array.max():.4f} V",
            f"φ range: {phi_array.max()-phi_array.min():.4f} V",
            f"J min: {j_array.min():.4e} A/m²",
            f"J max: {j_array.max():.4e} A/m²",
            f"J std: {j_array.std():.4e} A/m²",
        ])
        
        ax4.text(0.05, 0.95, '\n'.join(info_text), 
                transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{prefix}_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  → Results saved:")
        print(f"     Data: {data_file}")
        print(f"     Summary: {summary_file}")
        print(f"     Plot: {plot_file}")
    
    def retrain_model(self):
        """Retrain BNN with accumulated new samples"""
        print(f"\n{'='*70}")
        print(f"RETRAINING TRIGGERED")
        print(f"{'='*70}")
        print(f"New samples: {len(self.new_samples)}")
        
        if self.dataset is None:
            print("⚠ No base dataset - cannot retrain")
            return
        
        # Merge new samples with existing dataset
        new_inputs = np.array([s['input'] for s in self.new_samples])
        new_outputs = np.array([s['output'] for s in self.new_samples])
        new_metadata = [s['metadata'] for s in self.new_samples]
        
        updated_inputs = np.vstack([self.dataset['inputs'], new_inputs])
        updated_outputs = np.vstack([self.dataset['outputs'], new_outputs])
        updated_metadata = self.dataset['metadata'] + new_metadata
        
        updated_dataset = {
            'inputs': updated_inputs,
            'outputs': updated_outputs,
            'metadata': updated_metadata
        }
        
        print(f"Dataset size: {len(self.dataset['inputs'])} → {len(updated_inputs)}")
        
        # Save updated dataset
        dataset_backup = f"{self.dataset_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up dataset: {dataset_backup}")
        with open(dataset_backup, 'wb') as f:
            pickle.dump(self.dataset, f)
        
        print(f"Saving updated dataset: {self.dataset_path}")
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(updated_dataset, f)
        
        # Backup current model
        model_backup = f"{self.model_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up model: {model_backup}")
        os.system(f"cp {self.model_path} {model_backup}")
        
        # Retrain with more iterations for fine-tuning
        print(f"\nRetraining BNN (5000 iterations for fine-tuning)...")
        train_bnn_batch(
            dataset=updated_dataset,
            model_path=self.model_path,
            num_iterations=5000,
            learning_rate=0.001,  # Lower LR for fine-tuning
            hidden_dims=[64, 128, 64]
        )
        
        # Reload updated model
        print("Reloading updated model...")
        self.bnn = BNNWrapper.load(self.model_path)
        self.dataset = updated_dataset
        
        # Clear new samples
        self.new_samples = []
        self.stats['retrains'] += 1
        
        print(f"\n✓ Retraining complete!")
        print(f"{'='*70}\n")
    
    def print_statistics(self):
        """Print usage statistics"""
        print(f"\n{'='*70}")
        print("ACTIVE LEARNING STATISTICS")
        print(f"{'='*70}")
        print(f"Total predictions:     {self.stats['total_predictions']}")
        print(f"BNN used:              {self.stats['bnn_used']} "
              f"({100*self.stats['bnn_used']/max(1, self.stats['total_predictions']):.1f}%)")
        print(f"Physics used:          {self.stats['physics_used']} "
              f"({100*self.stats['physics_used']/max(1, self.stats['total_predictions']):.1f}%)")
        print(f"Retraining events:     {self.stats['retrains']}")
        
        if self.stats['avg_uncertainty']:
            avg_unc = np.mean(self.stats['avg_uncertainty'])
            print(f"Avg BNN uncertainty:   {avg_unc*100:.2f}%")
        
        # Calculate speedup
        if self.stats['physics_used'] > 0:
            physics_time = self.stats['physics_used'] * 6.5  # 6.5 min per physics sim
            bnn_time = self.stats['bnn_used'] * 0.002  # ~0.1s = 0.002 min per BNN
            total_time = physics_time + bnn_time
            all_physics_time = self.stats['total_predictions'] * 6.5
            speedup = all_physics_time / total_time
            
            print(f"\nTime savings:")
            print(f"  All physics:         {all_physics_time:.1f} min")
            print(f"  Active learning:     {total_time:.1f} min")
            print(f"  Speedup:             {speedup:.1f}x")
        
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Active learning pipeline for BNN corrosion predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction with active learning:
  python3 active_learning.py --model bnn_model.pt --data training_data.pkl \\
      --nacl 0.3 --temp 290 --ph 7.0 --flow 2.5
  
  # Batch predictions from file:
  python3 active_learning.py --model bnn_model.pt --data training_data.pkl \\
      --params new_conditions.txt --retrain-every 10
  
  # Adjust uncertainty threshold:
  python3 active_learning.py --model bnn_model.pt --data training_data.pkl \\
      --params batch.txt --uncertainty 0.03  # 3% threshold (more conservative)

Parameter file format (CSV):
  # NaCl(M), Temp(K), pH, Flow(m/s)
  0.3, 290, 7.0, 2.5
  0.8, 300, 8.5, 1.0
        """)
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained BNN model')
    
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training dataset (required for retraining)')
    
    parser.add_argument('--params', type=str, default=None,
                       help='Path to parameter file (CSV format)')
    
    # Single prediction parameters
    parser.add_argument('--nacl', type=float, default=None)
    parser.add_argument('--temp', type=float, default=None)
    parser.add_argument('--ph', type=float, default=None)
    parser.add_argument('--flow', type=float, default=None)
    
    parser.add_argument('--uncertainty', type=float, default=0.05,
                       help='Uncertainty threshold (default: 0.05 = 5%%)')
    
    parser.add_argument('--retrain-every', type=int, default=1,
                       help='Retrain after N new samples (default: 1)')
    
    parser.add_argument('--force-physics', action='store_true',
                       help='Force physics simulation (for testing)')
    
    parser.add_argument('--dependency-root', type=str,
                       default='../corrosion-modeling-applications',
                       help='Path to physics simulation code')
    
    parser.add_argument('--output', '-o', type=str, default='active_learning_results',
                       help='Output directory for results and plots (default: active_learning_results)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(
        model_path=args.model,
        dataset_path=args.data,
        uncertainty_threshold=args.uncertainty,
        retrain_every=args.retrain_every,
        dependency_root=args.dependency_root,
        output_dir=args.output
    )
    
    # Run predictions
    if args.params:
        # Batch mode
        print(f"Loading parameters from: {args.params}")
        params_list = []
        
        with open(args.params, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                values = [float(x) for x in line.split(',')]
                if len(values) != 4:
                    print(f"WARNING: Skipping invalid line: {line}")
                    continue
                
                params_list.append({
                    'NaCl': values[0],
                    'Temp': values[1],
                    'pH': values[2],
                    'Flow': values[3]
                })
        
        print(f"Loaded {len(params_list)} parameter sets\n")
        
        results = []
        for params in params_list:
            result = pipeline.predict(params, force_physics=args.force_physics)
            results.append(result)
        
        pipeline.print_statistics()
        
    elif all([args.nacl is not None, args.temp is not None, 
              args.ph is not None, args.flow is not None]):
        # Single prediction
        params = {
            'NaCl': args.nacl,
            'Temp': args.temp,
            'pH': args.ph,
            'Flow': args.flow
        }
        
        result = pipeline.predict(params, force_physics=args.force_physics)
        pipeline.print_statistics()
        
    else:
        print("ERROR: Must specify either:")
        print("  1. --params <file.txt>")
        print("  2. --nacl --temp --ph --flow (all four required)")
        sys.exit(1)


if __name__ == "__main__":
    main()
