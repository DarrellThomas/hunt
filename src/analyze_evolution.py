"""
Analyze evolutionary trends and speciation in the HUNT ecosystem.
Visualizes swim speed evolution and land vs water specialization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stats(filename='stats_autosave.npz'):
    """Load statistics from file."""
    if not Path(filename).exists():
        print(f"Error: {filename} not found. Run a simulation first!")
        return None

    data = np.load(filename)
    return {key: data[key] for key in data.files}

def analyze_speciation(stats):
    """Analyze if speciation (land vs water) is occurring."""
    timesteps = stats['timesteps']

    # Check if standard deviation is increasing (indicates divergence)
    prey_std = stats['prey_std_swim']
    pred_std = stats['pred_std_swim']

    # Simple heuristic: if std > initial_std * 1.5, speciation is occurring
    initial_prey_std = prey_std[0] if len(prey_std) > 0 else 0
    initial_pred_std = pred_std[0] if len(pred_std) > 0 else 0

    final_prey_std = prey_std[-1] if len(prey_std) > 0 else 0
    final_pred_std = pred_std[-1] if len(pred_std) > 0 else 0

    prey_speciation = final_prey_std > initial_prey_std * 1.5
    pred_speciation = final_pred_std > initial_pred_std * 1.5

    return {
        'prey_speciation': prey_speciation,
        'pred_speciation': pred_speciation,
        'prey_diversity_increase': (final_prey_std / initial_prey_std) if initial_prey_std > 0 else 1.0,
        'pred_diversity_increase': (final_pred_std / initial_pred_std) if initial_pred_std > 0 else 1.0,
    }

def plot_evolution(stats):
    """Create comprehensive evolution visualization."""
    timesteps = stats['timesteps']

    # Check if we have swim speed data (new format) or not (old format)
    has_swim_data = 'prey_avg_swim' in stats

    if has_swim_data:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle('HUNT Ecosystem - Evolutionary Dynamics', fontsize=16, fontweight='bold')

    # Plot 1: Population over time
    ax = axes[0, 0]
    ax.plot(timesteps, stats['prey_count'], 'g-', label='Prey', linewidth=2)
    ax.plot(timesteps, stats['pred_count'], 'r-', label='Predators', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_swim_data:
        # Plot 2: Swim speed evolution (average)
        ax = axes[0, 1]
        ax.plot(timesteps, stats['prey_avg_swim'], 'g-', label='Prey Avg', linewidth=2)
        ax.plot(timesteps, stats['pred_avg_swim'], 'r-', label='Predator Avg', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Swim Speed')
        ax.set_title('Swim Speed Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Swim speed diversity (std dev)
        ax = axes[1, 0]
        ax.plot(timesteps, stats['prey_std_swim'], 'g-', label='Prey Std Dev', linewidth=2)
        ax.plot(timesteps, stats['pred_std_swim'], 'r-', label='Predator Std Dev', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Swim Speed Std Dev')
        ax.set_title('Diversity in Swimming Ability (Higher = More Speciation)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Habitat preference (% in river)
        ax = axes[1, 1]
        ax.plot(timesteps, stats['prey_in_river_pct'], 'g-', label='Prey in River', linewidth=2)
        ax.plot(timesteps, stats['pred_in_river_pct'], 'r-', label='Predators in River', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('% of Population in River')
        ax.set_title('Habitat Distribution (Land vs Water)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% line')

        # Plot 5: Average age (proxy for fitness)
        ax = axes[2, 0]
        ax.plot(timesteps, stats['prey_avg_age'], 'g-', label='Prey', linewidth=2)
        ax.plot(timesteps, stats['pred_avg_age'], 'r-', label='Predators', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Age')
        ax.set_title('Average Lifespan (Fitness Proxy)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Speciation analysis text summary
        ax = axes[2, 1]
        ax.axis('off')

        # Analyze speciation
        analysis = analyze_speciation(stats)
    else:
        # Old format - just show basic stats
        # Plot 2: Average age
        ax = axes[0, 1]
        ax.plot(timesteps, stats['prey_avg_age'], 'g-', label='Prey', linewidth=2)
        ax.plot(timesteps, stats['pred_avg_age'], 'r-', label='Predators', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Age')
        ax.set_title('Average Lifespan (Fitness Proxy)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Predator energy
        ax = axes[1, 0]
        ax.plot(timesteps, stats['pred_avg_energy'], 'r-', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Energy')
        ax.set_title('Predator Energy (Hunting Success)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Message about old format
        ax = axes[1, 1]
        ax.axis('off')

        message = "OLD DATA FORMAT DETECTED\n" + "="*40 + "\n\n"
        message += "This stats file was generated before\n"
        message += "swim speed tracking was added.\n\n"
        message += "To see full evolutionary analysis:\n"
        message += "1. Run a new simulation\n"
        message += "2. It will auto-save new format data\n"
        message += "3. Run this analysis script again\n\n"
        message += "New features you'll see:\n"
        message += "• Swim speed evolution\n"
        message += "• Speciation detection\n"
        message += "• Habitat preference (river/island/land)\n"
        message += "• Evolutionary branching analysis"

        ax.text(0.05, 0.95, message, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        analysis = None

    if analysis is not None:
        summary_text = "EVOLUTIONARY ANALYSIS\n" + "="*40 + "\n\n"

        # Prey analysis
        summary_text += "PREY:\n"
        if analysis['prey_speciation']:
            summary_text += "  ✓ SPECIATION DETECTED!\n"
            summary_text += f"  • Diversity increased {analysis['prey_diversity_increase']:.1f}x\n"
        else:
            summary_text += "  ✗ No clear speciation yet\n"
            summary_text += f"  • Diversity change: {analysis['prey_diversity_increase']:.1f}x\n"

        prey_river_pct = stats['prey_in_river_pct'][-1] if len(stats['prey_in_river_pct']) > 0 else 0
        if prey_river_pct < 30:
            summary_text += "  • Preference: LAND specialists\n"
        elif prey_river_pct > 70:
            summary_text += "  • Preference: WATER specialists\n"
        else:
            summary_text += "  • Preference: Mixed/Both habitats\n"

        summary_text += f"  • {prey_river_pct:.1f}% in river\n\n"

        # Predator analysis
        summary_text += "PREDATORS:\n"
        if analysis['pred_speciation']:
            summary_text += "  ✓ SPECIATION DETECTED!\n"
            summary_text += f"  • Diversity increased {analysis['pred_diversity_increase']:.1f}x\n"
        else:
            summary_text += "  ✗ No clear speciation yet\n"
            summary_text += f"  • Diversity change: {analysis['pred_diversity_increase']:.1f}x\n"

        pred_river_pct = stats['pred_in_river_pct'][-1] if len(stats['pred_in_river_pct']) > 0 else 0
        if pred_river_pct < 30:
            summary_text += "  • Preference: LAND specialists\n"
        elif pred_river_pct > 70:
            summary_text += "  • Preference: WATER specialists\n"
        else:
            summary_text += "  • Preference: Mixed/Both habitats\n"

        summary_text += f"  • {pred_river_pct:.1f}% in river\n\n"

        # Overall summary
        summary_text += "INTERPRETATION:\n"
        if analysis['prey_speciation'] or analysis['pred_speciation']:
            summary_text += "Evolutionary branching is occurring!\n"
            summary_text += "Some lineages are adapting to water,\n"
            summary_text += "others to land. This is speciation in action!"
        else:
            summary_text += "Population is still homogenizing.\n"
            summary_text += "Run longer to see speciation emerge."

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('evolution_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Evolution analysis saved to: evolution_analysis.png")
    plt.show()

def main():
    """Main analysis function."""
    print("\n" + "="*60)
    print("HUNT Ecosystem - Evolutionary Analysis")
    print("="*60 + "\n")

    stats = load_stats('stats_autosave.npz')
    if stats is None:
        return

    print(f"Loaded {len(stats['timesteps'])} data points")
    print(f"Timesteps: {stats['timesteps'][0]} to {stats['timesteps'][-1]}")
    print()

    # Analyze and plot
    plot_evolution(stats)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
