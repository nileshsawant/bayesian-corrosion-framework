#!/usr/bin/env python3
"""
Generate a high-quality flowchart for the active learning framework
suitable for SERDP pre-proposal submission.

Requirements: pip install graphviz
"""

from graphviz import Digraph

def create_flowchart():
    """Create detailed active learning flowchart."""
    
    # Create graph with better styling for publication
    dot = Digraph(comment='Active Learning Framework',
                  format='png',
                  graph_attr={
                      'rankdir': 'TB',
                      'dpi': '300',
                      'bgcolor': 'white',
                      'fontname': 'Arial',
                      'fontsize': '12',
                      'splines': 'ortho',
                      'nodesep': '0.5',
                      'ranksep': '0.8'
                  },
                  node_attr={
                      'fontname': 'Arial',
                      'fontsize': '11',
                      'style': 'filled',
                      'shape': 'box',
                      'margin': '0.2,0.1'
                  },
                  edge_attr={
                      'fontname': 'Arial',
                      'fontsize': '10',
                      'penwidth': '2.0'
                  })
    
    # Start/End nodes
    dot.node('start', 'User Inputs\nEnvironmental Parameters\n(NaCl, T, pH, Flow)',
             shape='ellipse', fillcolor='#e1f5e1', width='2.5')
    dot.node('end', 'Results Delivered\nModel Self-Improved',
             shape='ellipse', fillcolor='#e1f5e1', width='2.5')
    
    # Input processing
    dot.node('input', 'Environmental Conditions\n\n• NaCl: 0.1-1.0 M\n• Temp: 278-313 K\n• pH: 6.0-9.0\n• Flow: 0.1-3.0 m/s',
             fillcolor='#ffffff', shape='box', style='filled')
    
    # BNN prediction
    dot.node('bnn', 'BNN Prediction\n\n• Forward pass: ~0.1 sec\n• 1000 posterior samples\n• Full φ field (2541 pts)\n• Full J profile (60 pts)',
             fillcolor='#d1ecf1', width='2.8')
    
    # Uncertainty calculation
    dot.node('uncert', 'Calculate Uncertainties\n\nφ_unc = σ_φ / mean(|φ|)\nJ_unc = σ_J / mean(|J|)\n\nCombined = max(φ_unc, J_unc)',
             fillcolor='#fff3cd', width='2.8')
    
    # Decision node
    dot.node('decision', 'Combined\nUncertainty\n> Threshold?\n(default: 5%)',
             shape='diamond', fillcolor='#fff4e6', width='2.2', height='1.5')
    
    # Fast path (BNN)
    dot.node('fast', 'Return BNN Prediction\n\n✓ 400-1000× Speedup\n✓ Full uncertainty bands (±2σ)\n✓ φ and J profiles\n\nTypical: 0.17% φ, 1.95% J uncertainty',
             fillcolor='#d4edda', width='3.0')
    
    # Physics path
    dot.node('physics', 'Run Physics Simulation\n\n• Octave/MATLAB solver\n• Electrochemical model\n• Full geometry resolution\n• Time: ~6.5 minutes',
             fillcolor='#f8d7da', width='2.8')
    
    # Add to dataset
    dot.node('adddata', 'Add to Training Dataset\n\n• Automatic backup created\n• Dataset: N → N+1 samples\n• Preserves old versions',
             fillcolor='#e7e7e7', width='2.8')
    
    # Retrain
    dot.node('retrain', 'Retrain BNN Model\n\n• Adaptive iterations:\n  - N≤30: 3000 iterations\n  - 31-50: 5000 iterations\n  - N>50: 8000 iterations\n• Separate σ_φ, σ_J learning\n• Time: 3-8 min (H100 GPU)',
             fillcolor='#cce5ff', width='3.0')
    
    # Update model
    dot.node('update', 'Update Model Weights\n\n• Save new bnn_model.pt\n• Backup old model\n• Improved coverage',
             fillcolor='#e7e7e7', width='2.8')
    
    # Physics result
    dot.node('physresult', 'Return Physics Result\n\n✓ High accuracy ground truth\n✓ Model now improved\n✓ Future similar queries: fast BNN',
             fillcolor='#d4edda', width='3.0')
    
    # Save results
    dot.node('save', 'Save Results\n\n• 4-panel plot with ±2σ bands\n• CSV statistics\n• Pickle data for reuse',
             fillcolor='#e7e7e7', width='2.8')
    
    # Create edges
    dot.edge('start', 'input')
    dot.edge('input', 'bnn')
    dot.edge('bnn', 'uncert')
    dot.edge('uncert', 'decision')
    
    # Decision branches
    dot.edge('decision', 'fast', label='  Low Uncertainty\n  HIGH CONFIDENCE  ',
             color='#28a745', penwidth='3.0', fontcolor='#28a745')
    dot.edge('decision', 'physics', label='  High Uncertainty\n  NEEDS VALIDATION  ',
             color='#dc3545', penwidth='3.0', fontcolor='#dc3545')
    
    # Fast path
    dot.edge('fast', 'save')
    
    # Physics path
    dot.edge('physics', 'adddata')
    dot.edge('adddata', 'retrain')
    dot.edge('retrain', 'update')
    dot.edge('update', 'physresult')
    dot.edge('physresult', 'save')
    
    # End
    dot.edge('save', 'end')
    
    return dot

def create_simple_flowchart():
    """Create simplified flowchart for presentations."""
    
    dot = Digraph(comment='Active Learning Framework - Simple',
                  format='png',
                  graph_attr={
                      'rankdir': 'TB',
                      'dpi': '300',
                      'bgcolor': 'white',
                      'fontname': 'Arial',
                      'fontsize': '14',
                      'splines': 'polyline',
                      'nodesep': '0.6',
                      'ranksep': '0.7'
                  },
                  node_attr={
                      'fontname': 'Arial Bold',
                      'fontsize': '13',
                      'style': 'filled,rounded',
                      'shape': 'box',
                      'margin': '0.3,0.2',
                      'penwidth': '2.5'
                  },
                  edge_attr={
                      'fontname': 'Arial',
                      'fontsize': '11',
                      'penwidth': '2.5'
                  })
    
    # Nodes with larger text
    dot.node('1', 'User Inputs\n(NaCl, T, pH, Flow)',
             shape='ellipse', fillcolor='#90EE90', width='2.5', height='1.0')
    
    dot.node('2', 'BNN Prediction\n(~0.1 seconds)',
             fillcolor='#87CEEB', width='2.5')
    
    dot.node('3', 'Uncertainty\n> 5%?',
             shape='diamond', fillcolor='#FFD700', width='2.5', height='1.3')
    
    dot.node('4', 'Fast BNN Result\n400-1000× Speedup',
             fillcolor='#98FB98', width='2.5')
    
    dot.node('5', 'Physics Simulation\n(~6.5 minutes)',
             fillcolor='#FFB6C6', width='2.5')
    
    dot.node('6', 'Retrain Model\n(3-8 minutes)',
             fillcolor='#ADD8E6', width='2.5')
    
    dot.node('7', 'Physics Result\n(Model Improved)',
             fillcolor='#98FB98', width='2.5')
    
    dot.node('8', 'Save Results',
             shape='ellipse', fillcolor='#D3D3D3', width='2.5', height='1.0')
    
    # Edges
    dot.edge('1', '2', penwidth='3')
    dot.edge('2', '3', penwidth='3')
    dot.edge('3', '4', label=' NO ', color='green', fontcolor='green', penwidth='3')
    dot.edge('3', '5', label=' YES ', color='red', fontcolor='red', penwidth='3')
    dot.edge('4', '8', penwidth='3')
    dot.edge('5', '6', penwidth='3')
    dot.edge('6', '7', penwidth='3')
    dot.edge('7', '8', penwidth='3')
    
    return dot

if __name__ == '__main__':
    print("Generating flowcharts...")
    
    # Detailed flowchart
    detailed = create_flowchart()
    detailed.render('active_learning_flowchart_detailed', cleanup=True)
    print("✓ Created: active_learning_flowchart_detailed.png")
    
    # Simple flowchart
    simple = create_simple_flowchart()
    simple.render('active_learning_flowchart_simple', cleanup=True)
    print("✓ Created: active_learning_flowchart_simple.png")
    
    print("\nFlowcharts ready for SERDP pre-proposal!")
    print("- Use 'detailed' version for technical documentation")
    print("- Use 'simple' version for executive summary or presentations")
