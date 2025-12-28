import matplotlib.pyplot as plt
import numpy as np
import os


def plot_global_constraints(history, global_constraints, save_path=None):
    """
    Plot global constraint satisfaction over epochs.

    Args:
        history: Dict with 'epochs' and 'global_predictions' keys
        global_constraints: List/array of constraint values per class
        save_path: Optional path to save the figure
    """
    epochs = history['epochs']
    predictions = history['global_predictions']  # Shape: (n_epochs, n_classes)

    fig, ax = plt.subplots(figsize=(12, 6))

    class_names = ['Dropout', 'Enrolled', 'Graduate']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Plot predicted counts for each class
    for class_id in range(3):
        class_preds = [pred[class_id] for pred in predictions]
        ax.plot(epochs, class_preds, label=f'{class_names[class_id]} (Predicted)',
                color=colors[class_id], linewidth=2)

        # Plot constraint as horizontal dotted line
        constraint = global_constraints[class_id]
        if constraint < 1e9:  # Only plot if constrained
            ax.axhline(y=constraint, color=colors[class_id], linestyle='--',
                      linewidth=2, alpha=0.7,
                      label=f'{class_names[class_id]} Constraint ({int(constraint)})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Predicted Students', fontsize=12)
    ax.set_title('Global Constraint Satisfaction Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Global constraints plot saved to: {save_path}")

    plt.close()


def plot_local_constraints(history, local_constraints, save_path=None, max_courses=6):
    """
    Plot local constraint satisfaction over epochs for the tracked course.

    Args:
        history: Dict with 'epochs' and 'local_predictions' keys
        local_constraints: Dict mapping course_id -> [constraint0, constraint1, constraint2]
        save_path: Optional path to save the figure
        max_courses: Unused (kept for compatibility)
    """
    from config.experiment_config import TRACKED_COURSE_ID

    epochs = history['epochs']
    predictions = history['local_predictions']  # List of dicts: [{course_id: [c0, c1, c2]}, ...]

    # Plot only the tracked course
    if TRACKED_COURSE_ID not in local_constraints:
        print(f"Warning: Tracked course {TRACKED_COURSE_ID} not found in local constraints. Skipping local constraint plot.")
        return

    # Single plot for tracked course
    fig, ax = plt.subplots(figsize=(12, 6))

    class_names = ['Dropout', 'Enrolled', 'Graduate']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    constraints = local_constraints[TRACKED_COURSE_ID]

    # Extract predictions for this course over epochs
    for class_id in range(3):
        class_preds = []
        for epoch_preds in predictions:
            if TRACKED_COURSE_ID in epoch_preds:
                class_preds.append(epoch_preds[TRACKED_COURSE_ID][class_id])
            else:
                class_preds.append(0)

        ax.plot(epochs, class_preds, label=f'{class_names[class_id]} (Predicted)',
               color=colors[class_id], linewidth=2)

        # Plot constraint line
        if constraints[class_id] < 1e9:
            ax.axhline(y=constraints[class_id], color=colors[class_id],
                      linestyle='--', linewidth=2, alpha=0.7,
                      label=f'{class_names[class_id]} Constraint ({int(constraints[class_id])})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Predicted Students', fontsize=12)
    ax.set_title(f'Local Constraint Satisfaction Over Training (Course {TRACKED_COURSE_ID})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Local constraints plot saved to: {save_path}")

    plt.close()


def plot_losses(history, save_path=None):
    """
    Plot loss components over epochs.

    Args:
        history: Dict with 'epochs', 'loss_global', 'loss_local', 'loss_ce' keys
        save_path: Optional path to save the figure
    """
    epochs = history['epochs']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # L_target (global)
    axes[0, 0].plot(epochs, history['loss_global'], color='#e74c3c', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('L_target (Global Constraint Loss)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1e-6, color='green', linestyle='--', alpha=0.5, label='Threshold')
    axes[0, 0].legend()

    # L_feat (local)
    axes[0, 1].plot(epochs, history['loss_local'], color='#3498db', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('L_feat (Local Constraint Loss)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1e-6, color='green', linestyle='--', alpha=0.5, label='Threshold')
    axes[0, 1].legend()

    # L_pred (CE)
    axes[1, 0].plot(epochs, history['loss_ce'], color='#f39c12', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('L_pred (Cross-Entropy Loss)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # L_total (computed from components)
    loss_total = []
    for i in range(len(epochs)):
        total = history['loss_ce'][i] + history['lambda_global'][i] * history['loss_global'][i] + history['lambda_local'][i] * history['loss_local'][i]
        loss_total.append(total)
    axes[1, 1].plot(epochs, loss_total, color='#9b59b6', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('L_total (Combined Loss)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Loss Components Over Training', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")

    plt.close()


def plot_lambda_evolution(history, save_path=None):
    """
    Plot lambda weight evolution over epochs.

    Args:
        history: Dict with 'epochs', 'lambda_global', 'lambda_local' keys
        save_path: Optional path to save the figure
    """
    if 'lambda_global' not in history or len(history['lambda_global']) == 0:
        return

    epochs = history['epochs']
    lambda_global = history['lambda_global']
    lambda_local = history['lambda_local']

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, lambda_global, label='λ_global (Target Constraint)',
            color='#e74c3c', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, lambda_local, label='λ_local (Feature Constraint)',
            color='#3498db', linewidth=2, marker='s', markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Lambda Weight', fontsize=12)
    ax.set_title('Adaptive Lambda Weight Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotations for significant changes
    if len(lambda_global) > 1:
        max_global = max(lambda_global)
        max_idx = lambda_global.index(max_global)
        ax.annotate(f'Max: {max_global:.1f}',
                   xy=(epochs[max_idx], max_global),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='#e74c3c',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lambda evolution plot saved to: {save_path}")

    plt.close()


def create_all_visualizations(history, global_constraints, local_constraints, output_dir='./results'):
    """
    Create all training visualizations.

    Args:
        history: Training history dict
        global_constraints: Global constraint values
        local_constraints: Local constraint values dict
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("Creating Training Visualizations")
    print("="*80)

    # Global constraints plot
    plot_global_constraints(
        history,
        global_constraints,
        save_path=os.path.join(output_dir, 'global_constraints.png')
    )

    # Local constraints plot
    plot_local_constraints(
        history,
        local_constraints,
        save_path=os.path.join(output_dir, 'local_constraints.png')
    )

    # Loss plots
    plot_losses(
        history,
        save_path=os.path.join(output_dir, 'losses.png')
    )

    # Lambda evolution plot
    plot_lambda_evolution(
        history,
        save_path=os.path.join(output_dir, 'lambda_evolution.png')
    )

    print("="*80)
    print("All visualizations created successfully!")
    print("="*80 + "\n")
