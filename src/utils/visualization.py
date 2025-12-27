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
    Plot local constraint satisfaction over epochs for selected courses.

    Args:
        history: Dict with 'epochs' and 'local_predictions' keys
        local_constraints: Dict mapping course_id -> [constraint0, constraint1, constraint2]
        save_path: Optional path to save the figure
        max_courses: Maximum number of courses to plot (to avoid clutter)
    """
    epochs = history['epochs']
    predictions = history['local_predictions']  # List of dicts: [{course_id: [c0, c1, c2]}, ...]

    # Select courses that have violations most often (most interesting to plot)
    course_ids = list(local_constraints.keys())

    # If too many courses, select subset
    if len(course_ids) > max_courses:
        # Compute violation frequency for each course
        violation_counts = {cid: 0 for cid in course_ids}
        for epoch_preds in predictions:
            for course_id in course_ids:
                if course_id in epoch_preds:
                    preds = epoch_preds[course_id]
                    constraints = local_constraints[course_id]
                    for class_id in range(3):
                        if constraints[class_id] < 1e9 and preds[class_id] > constraints[class_id]:
                            violation_counts[course_id] += 1

        # Select courses with most violations
        course_ids = sorted(violation_counts.keys(), key=lambda x: violation_counts[x], reverse=True)[:max_courses]

    # Create subplots: one for each course
    n_courses = len(course_ids)
    n_cols = 2
    n_rows = (n_courses + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    class_names = ['Dropout', 'Enrolled', 'Graduate']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, course_id in enumerate(course_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        constraints = local_constraints[course_id]

        # Extract predictions for this course over epochs
        for class_id in range(3):
            class_preds = []
            for epoch_preds in predictions:
                if course_id in epoch_preds:
                    class_preds.append(epoch_preds[course_id][class_id])
                else:
                    class_preds.append(0)

            ax.plot(epochs, class_preds, label=class_names[class_id],
                   color=colors[class_id], linewidth=1.5)

            # Plot constraint line
            if constraints[class_id] < 1e9:
                ax.axhline(y=constraints[class_id], color=colors[class_id],
                          linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_title(f'Course {course_id}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Predicted Students', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(course_ids), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle('Local Constraint Satisfaction Over Training (Selected Courses)',
                 fontsize=14, fontweight='bold', y=1.00)
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
