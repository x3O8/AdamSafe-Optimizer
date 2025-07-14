import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class AdamSafe(optim.Optimizer):
    """
    Implements a custom Adam-like optimizer with a Gradient Stability Factor (GSF).
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_gsf=0.9, eps=1e-8, weight_decay=0, clip_norm=1.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_gsf=beta_gsf, eps=eps, weight_decay=weight_decay, clip_norm=clip_norm)
        super(AdamSafe, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad.clamp_(-group['clip_norm'], group['clip_norm'])
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['gsf'] = torch.tensor(1.0, device=p.device)
                    state['prev_grad_norm'] = torch.tensor(0.0, device=p.device)

                m, s = state['m'], state['s']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                s.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                grad_norm = torch.norm(grad)
                prev_grad_norm = state['prev_grad_norm']

                if grad_norm > 0 and prev_grad_norm > 0:
                    norm_ratio = grad_norm / (prev_grad_norm + group['eps'])
                    gsf_t = 1.0 / (1.0 + torch.abs(norm_ratio - 1.0))
                else:
                    gsf_t = torch.tensor(0.5, device=p.device)

                state['gsf'].mul_(group['beta_gsf']).add_(gsf_t.clamp(0.2, 1.0), alpha=1 - group['beta_gsf'])
                state['prev_grad_norm'] = grad_norm

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_hat = m / bias_correction1
                s_hat = s / bias_correction2

                step_size = group['lr'] * state['gsf']
                p.addcdiv_(m_hat, (s_hat.sqrt().add_(group['eps'])), value=-step_size)

class AdaBelief(optim.Optimizer):
    """
    Implements the AdaBelief optimizer.
    Reference: https://github.com/juntang-zhuang/Adabelief-Optimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0,
                 amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, weight_decouple=weight_decouple,
                        fixed_decay=fixed_decay, rectify=rectify)
        super(AdaBelief, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if group['rectify']:
                    rho_t = 2 / (1 - beta2) - 1
                    rho_inf = rho_t - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step'])
                    if rho_inf >= 5:
                        r_t = math.sqrt(((rho_inf - 4) * (rho_inf - 2) * rho_t) / ((rho_t - 4) * (rho_t - 2) * rho_inf))
                        p.addcdiv_(exp_avg, denom, value=-step_size * r_t)
                    else:
                        p.add_(exp_avg, alpha=-step_size)
                else:
                     p.addcdiv_(exp_avg, denom, value=-step_size)


def get_dataloaders(dataset_name='CIFAR-10', batch_size=128):
    """Prepares data loaders for specified dataset."""
    if dataset_name == 'CIFAR-10':
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        dataset_class = datasets.CIFAR10
        num_classes = 10
    elif dataset_name == 'CIFAR-100':
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        dataset_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError("Dataset not supported. Choose 'CIFAR-10' or 'CIFAR-100'.")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = dataset_class(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=test_transform)

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    return train_loader, test_loader, num_classes

def get_model(num_classes=10):
    """Returns a pretrained ResNet-18 model with a modified classifier."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def run_training_benchmark(optimizer_class, optimizer_params, dataset_name, num_epochs=10):
    """Trains a model and returns performance metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, num_classes = get_dataloaders(dataset_name)
    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    print(f"--- Training {optimizer_class.__name__} on {dataset_name} for {num_epochs} epochs ---")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_corrects = 0
        num_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item() * images.size(0)
            epoch_corrects += torch.sum(preds == labels.data)
            num_samples += len(labels)

        print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {epoch_loss/num_samples:.4f}, Train Acc: {epoch_corrects.double()/num_samples:.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    model.eval()
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "time": total_time
    }

def beale_function(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

def rosenbrock_function(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def func_a(x, y): return torch.abs(x) + torch.abs(y)
def func_c(x, y): return (x + y)**2 + (x - y)**2 / 10.0

def get_optimizer_trajectory(optimizer_class, func, init_point, lr=0.01, steps=500, **kwargs):
    point = torch.tensor(init_point, dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_class([point], lr=lr, **kwargs)
    path = np.zeros((steps + 1, 3))
    path[0] = (point.detach().numpy()[0], point.detach().numpy()[1], func(point[0], point[1]).item())

    for i in range(steps):
        optimizer.zero_grad()
        loss = func(point[0], point[1])
        loss.backward()
        optimizer.step()
        path[i+1] = (point.detach().numpy()[0], point.detach().numpy()[1], func(point[0], point[1]).item())
    return path

def plot_2d_trajectory(ax, func, paths, labels, title, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(torch.from_numpy(X), torch.from_numpy(Y)).numpy()
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', norm=LogNorm())
    for path, label in zip(paths, labels):
        ax.plot(path[:, 0], path[:, 1], 'o-', label=label, markersize=2, linewidth=1.5)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

def plot_3d_trajectory(ax, func, paths, labels, title, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(torch.from_numpy(X), torch.from_numpy(Y)).numpy()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, rstride=5, cstride=5)
    for path, label in zip(paths, labels):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'o-', label=label, linewidth=2, markersize=3, zorder=10)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x, y)'); ax.legend()


if __name__ == "__main__":
    # --- Set seed for reproducibility ---
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Control Flags ---
    RUN_DEEP_LEARNING_BENCHMARKS = True
    RUN_TRAJECTORY_ANALYSIS = True

    # --- 4.1 Deep Learning Benchmarks ---
    if RUN_DEEP_LEARNING_BENCHMARKS:
        print("\n" + "="*80)
        print("PART 1: DEEP LEARNING OPTIMIZER BENCHMARKS")
        print("="*80)

        datasets_to_test = ['CIFAR-10', 'CIFAR-100']
        optimizers_to_benchmark = {
            "Adam": (optim.Adam, {'lr': 1e-4}),
            "AdamSafe": (AdamSafe, {'lr': 1e-4, 'clip_norm': 1.0}),
            "AdaBelief": (AdaBelief, {'lr': 1e-4, 'eps': 1e-12, 'rectify': True})
        }

        for dataset in datasets_to_test:
            print(f"\n\n--- BENCHMARKING ON {dataset} ---")
            results = {}
            for name, (opt_class, params) in optimizers_to_benchmark.items():
                # Re-seed before each run for fair comparison
                torch.manual_seed(42)
                np.random.seed(42)
                results[name] = run_training_benchmark(opt_class, params, dataset_name=dataset, num_epochs=10)

            # Print results table for the dataset
            print("\n" + "-"*60)
            print(f"Results for {dataset}:")
            print(f"{'Optimizer':<12} | {'Accuracy':<10} | {'F1 Score':<10} | {'Time (s)':<10}")
            print("-" * 60)
            for name, metrics in results.items():
                print(f"{name:<12} | {metrics['accuracy']:.4f}     | {metrics['f1_score']:.4f}     | {metrics['time']:.2f}")
            print("-" * 60)

    # --- 4.2 Mathematical Function Trajectory Analysis ---
    if RUN_TRAJECTORY_ANALYSIS:
        print("\n" + "="*80)
        print("PART 2: OPTIMIZER TRAJECTORY ANALYSIS")
        print("="*80)

        optimizers_to_visualize = {
            "Adam": (optim.Adam, {'lr': 0.05}),
            "AdamSafe": (AdamSafe, {'lr': 0.05, 'clip_norm': 1.0}),
            "AdaBelief": (AdaBelief, {'lr': 0.05, 'eps': 1e-12, 'rectify': True})
        }

        test_functions = {
            "Beale Function": (beale_function, [-4.5, 4.5], [-4.5, 4.5], [-4.0, 4.0]),
            "Rosenbrock Function": (rosenbrock_function, [-2, 2], [-1, 3], [-1.5, 1.5]),
            "f(x,y) = |x| + |y|": (func_a, [-2, 2], [-2, 2], [-1.5, 1.5]),
            "f(x,y) = (x+y)^2 + (x-y)^2/10": (func_c, [-2, 2], [-2, 2], [-1.5, 1.5]),
        }

        for name, (func, x_range, y_range, init_point) in test_functions.items():
            print(f"\n--- Analyzing {name} ---")
            paths, labels = [], []
            for opt_name, (opt_class, params) in optimizers_to_visualize.items():
                path = get_optimizer_trajectory(opt_class, func, init_point, **params)
                paths.append(path)
                labels.append(opt_name)

            fig = plt.figure(figsize=(14, 6))
            fig.suptitle(f"Optimizer Comparison on {name}", fontsize=16)
            ax1 = fig.add_subplot(1, 2, 1)
            plot_2d_trajectory(ax1, func, paths, labels, "2D Trajectory", x_range, y_range)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            plot_3d_trajectory(ax2, func, paths, labels, "3D Trajectory", x_range, y_range)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
