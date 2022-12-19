from dataset import get_cifar10_dataset
from util import trainer, distill_trainer
from model import get_student, get_teacher
import argparse
import torch
from loss import KDLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='teacher', help='teacher/student')
    parser.add_argument('--lr_scratch', type=float, default=1e-2, help="Learning rate apply to retrain layers")
    parser.add_argument('--lr_finetune', type=float, default=1e-4, help="Learning rate apply to fine tuning layers")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")

    parser.add_argument('--temp', type=int, default=20, help="Temperature")
    parser.add_argument('--alpha', type=float, default=0.1, help="alpha")
    args = parser.parse_args()

    if args.mode == 'teacher':
        # teacher model
        model = get_teacher()

        # Criterion
        criterion = torch.nn.CrossEntropyLoss()

        # Parameter setting
        params_to_finetune = []
        params_from_scratch = []

        for name, param in model.named_parameters():
            if 'fc' not in name:
                params_to_finetune.append(param)
            
            else:
                params_from_scratch.append(param)

        # Optimizer
        optimizer = torch.optim.Adam([{'params' : params_to_finetune, 'lr':args.lr_finetune},
                                    {'params': params_from_scratch, 'lr':args.lr_scratch}],
                                    betas=(0.5, 0.999), weight_decay=5e-5, amsgrad=True)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # Dataloader
        batch_size = args.batch
        train_dataloader = torch.utils.data.DataLoader(get_cifar10_dataset('train')[0], batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(get_cifar10_dataset('test')[0], batch_size=batch_size, shuffle=False, num_workers=2)

        trainer(model, optimizer, scheduler, train_dataloader, test_dataloader, args.epochs, criterion, savedir="teacher/")


    elif args.mode == 'student':
        # student model
        teacher = get_teacher()
        student = get_student()

        # Criterion
        criterion = KDLoss(T=args.temp, alpha=args.alpha)

        # Optimizer
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr_scratch,
                                    betas=(0.5, 0.999),  weight_decay=5e-5, amsgrad=True)
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # Dataloader
        batch_size = args.batch
        train_dataloader = torch.utils.data.DataLoader(get_cifar10_dataset('train')[0], batch_size=batch_size, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(get_cifar10_dataset('test')[0], batch_size=batch_size, shuffle=False, num_workers=2)
        distill_trainer(student, teacher, optimizer, scheduler, train_dataloader, test_dataloader, args.epochs, criterion, "student", alpha=args.alpha)

    else:
        raise ValueError("Not implemented")