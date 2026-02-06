#!/usr/bin/env python3
"""
Systematic measurement of articulation behavior.

Maps:
1. Articulation rate across problem types
2. D_reaching activation vs articulation
3. Temperature effects on stochasticity
"""

import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load model and probe
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'checkpoints/stage1b/checkpoint-2000',
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)
model.eval()

print("Loading D_reaching probe...")
probes = torch.load('results/refined_probes.pt', weights_only=False)
D_reaching = probes['D_reaching'].to('cuda')
layer = probes['layer']

def measure_problem(a, b, n_samples=8, temperature=0.7):
    """Measure articulation rate and desire activation for a problem."""
    problem = f"{a} * {b}"
    prompt = f"""Solve: {problem}

You can use a calculator by writing:
<tool>calculator</tool><input>expression</input>

Or just give your answer directly."""
    
    messages = [{'role': 'user', 'content': prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    
    # Get desire activation (same for all samples)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1][0, -1, :].float()
        desire = torch.dot(hidden, D_reaching.float()).item()
    
    # Sample completions
    articulations = []
    for _ in range(n_samples):
        with torch.no_grad():
            gen = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
            out = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            articulations.append('<tool>' in out.lower())
    
    return {
        'a': a,
        'b': b,
        'answer': a * b,
        'a_digits': len(str(a)),
        'b_digits': len(str(b)),
        'desire_activation': desire,
        'articulation_rate': sum(articulations) / len(articulations),
        'articulations': articulations,
        'temperature': temperature,
    }

results = {
    'systematic_grid': [],
    'boundary_search': [],
    'temperature_sweep': [],
}

# 1. Systematic grid across digit combinations
print("\n=== Systematic Grid (8 samples each, temp=0.7) ===")
grid_problems = [
    # 1x1
    (3, 4), (7, 8), (9, 9),
    # 1x2
    (5, 23), (8, 67), (9, 99),
    # 2x2 small
    (12, 34), (23, 45), (34, 56),
    # 2x2 medium  
    (45, 67), (56, 78), (67, 89),
    # 2x2 large
    (78, 89), (89, 99), (99, 99),
    # 2x3
    (12, 345), (23, 456), (45, 678), (67, 890),
    # 3x3
    (123, 456), (234, 567), (456, 789),
]

for a, b in tqdm(grid_problems, desc="Grid"):
    r = measure_problem(a, b)
    results['systematic_grid'].append(r)
    rate = r['articulation_rate']
    desire = r['desire_activation']
    print(f"  {a:3d} x {b:3d} = {a*b:6d}  |  desire={desire:+6.1f}  |  art={rate:.0%}")

# 2. Fine-grained boundary search in 2x2 range
print("\n=== Boundary Search (2x2 range, 16 samples each) ===")
# Focus on the 56x78 to 67x89 range where we saw the switch
boundary_problems = [
    (56, 78), (58, 78), (60, 78), (62, 78), (64, 78), (66, 78), (68, 78),
    (56, 80), (56, 82), (56, 84), (56, 86), (56, 88),
    (60, 85), (62, 85), (64, 85), (66, 85),
    (65, 85), (65, 86), (65, 87), (65, 88), (65, 89),
]

for a, b in tqdm(boundary_problems, desc="Boundary"):
    r = measure_problem(a, b, n_samples=16)
    results['boundary_search'].append(r)
    rate = r['articulation_rate']
    desire = r['desire_activation']
    marker = "<<<" if 0 < rate < 1 else ""
    print(f"  {a:3d} x {b:3d} = {a*b:6d}  |  desire={desire:+6.1f}  |  art={rate:5.1%} {marker}")

# 3. Temperature sweep on a few problems
print("\n=== Temperature Sweep (16 samples each) ===")
temp_problems = [(56, 78), (67, 89), (60, 85)]
temperatures = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]

for a, b in temp_problems:
    print(f"\n  {a} x {b}:")
    for temp in temperatures:
        r = measure_problem(a, b, n_samples=16, temperature=temp)
        results['temperature_sweep'].append(r)
        rate = r['articulation_rate']
        marker = "<<<" if 0 < rate < 1 else ""
        print(f"    temp={temp:.1f}  ->  art={rate:5.1%} {marker}")

# Save results
output_path = Path('results/articulation_boundary_map.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

mixed = [r for r in results['systematic_grid'] + results['boundary_search'] 
         if 0 < r['articulation_rate'] < 1]
if mixed:
    print(f"\nFound {len(mixed)} problems with mixed articulation:")
    for r in mixed:
        print(f"  {r['a']} x {r['b']}: {r['articulation_rate']:.0%}")
else:
    print("\nNo problems found with mixed articulation (all 0% or 100%)")

# Correlation between desire and articulation
all_results = results['systematic_grid'] + results['boundary_search']
desires = [r['desire_activation'] for r in all_results]
arts = [r['articulation_rate'] for r in all_results]
correlation = sum((d - sum(desires)/len(desires)) * (a - sum(arts)/len(arts)) 
                  for d, a in zip(desires, arts))
correlation /= (len(desires) * (sum((d - sum(desires)/len(desires))**2 for d in desires)/len(desires))**0.5 
                * (sum((a - sum(arts)/len(arts))**2 for a in arts)/len(arts))**0.5)
print(f"\nCorrelation between desire activation and articulation rate: {correlation:.3f}")

