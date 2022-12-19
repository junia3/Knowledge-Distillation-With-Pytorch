import torch
import os
from tqdm import tqdm
import pickle5 as pickle
from loss import KDLoss

def trainer(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, loss_func, savedir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = {"train" : 0, "test" : 0}
    loss_hist = {"train" : [], "test" : []}
    acc_hist = {"train" : [], "test" : []}
    model = model.to(device)
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    else:
        raise ValueError("You need savedir")
        
    for epoch in range(epochs):
        training_loss, training_acc = 0.0, 0.0
        model.train()
        loading = tqdm(enumerate(train_dataloader), desc="training...")
        for i, (image, label) in loading:            
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            loss = loss_func(prediction, label)

            _, pred = torch.max(prediction, axis=1)
            
            training_loss += loss.item()

            training_acc += float(torch.sum(torch.eq(pred, label)))/len(pred)

            loss.backward()
            optimizer.step()
            
            loading.set_description(f"Training[{epoch+1}/{epochs}]... Loss : {training_loss/(i+1):.4f}, Acc : {100*training_acc/(i+1):.2f}%")
        
        loss_hist["train"] += [training_loss / len(train_dataloader)]
        acc_hist["train"] += [100*training_acc/len(train_dataloader)]
        
        # Update best accuracy
        if 100*training_acc/len(train_dataloader) > best_acc["train"]:
            best_acc["train"] = 100*training_acc/len(train_dataloader)

        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss, validation_acc = 0.0, 0.0
            loading = tqdm(enumerate(val_dataloader), desc="validating...")
            for i, (image, label) in loading:
                image, label = image.to(device), label.to(device)
                prediction = model(image)
                
                loss = loss_func(prediction, label)
                validation_loss += loss.item()

                _, pred = torch.max(prediction, axis=1)
                validation_acc += float(torch.sum(torch.eq(pred, label)))/len(pred)

                loading.set_description(f"Validating[{epoch+1}/{epochs}]... Loss : {validation_loss/(i+1):.4f}, Acc : {100*validation_acc/(i+1):.2f}%")
            
            loss_hist["test"] += [validation_loss / len(val_dataloader)]
            acc_hist["test"] += [100*validation_acc/len(val_dataloader)]
            
            # Update best accuracy
            if 100*validation_acc/len(val_dataloader) > best_acc["test"]:
                best_acc["test"] = 100*validation_acc/len(val_dataloader)     
                torch.save(
                    {
                        "model_state_dict": model.state_dict()
                    }
                    , os.path.join(savedir, "best.ckpt"))
                print("Best model so far!! --> Save file")
    
    results = {"best_acc" : best_acc, "loss_hist" : loss_hist, "acc_hist" : acc_hist}
    with open(os.path.join(savedir, "results.pickle"), "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(f"Best training acc : {best_acc['train']}",f"Best test acc : {best_acc['test']}")


def distill_trainer(student, teacher, optimizer, scheduler, train_dataloader, val_dataloader, epochs, loss_func, savedir, alpha=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = {"train" : 0, "test" : 0}
    loss_hist = {"train" : [], "test" : []}
    acc_hist = {"train" : [], "test" : []}
    
    # Student model
    student = student.to(device)
    test_criterion = KDLoss(T=1, alpha=alpha)

    # Teacher model -> get pre-trained network
    teacher = teacher.to(device)
    checkpoint = torch.load("teacher/best.ckpt")
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    else:
        raise ValueError("You need savedir")
        
    for epoch in range(epochs):
        # (0) : Training
        training_loss, training_acc = 0.0, 0.0
        student.train()
        loading = tqdm(enumerate(train_dataloader), desc="training...")
        for i, (image, label) in loading:
            
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            prediction = student(image)

            with torch.no_grad():
                teacher_prediction = teacher(image)

            loss = loss_func(prediction, label, teacher_prediction)
            _, pred = torch.max(prediction, axis=1)
            
            training_loss += loss.item()
            training_acc += float(torch.sum(torch.eq(pred, label)))/len(pred)
            loss.backward()
            optimizer.step()
            
            loading.set_description(f"Training[{epoch+1}/{epochs}]... Loss : {training_loss/(i+1):.4f}, Acc : {100*training_acc/(i+1):.2f}%")
        
        loss_hist["train"] += [training_loss / len(train_dataloader)]
        acc_hist["train"] += [100*training_acc/len(train_dataloader)]
        
        # Update best accuracy
        if 100*training_acc/len(train_dataloader) > best_acc["train"]:
            best_acc["train"] = 100*training_acc/len(train_dataloader)

        student.eval()
        with torch.no_grad():
            validation_loss, validation_acc = 0.0, 0.0
            loading = tqdm(enumerate(val_dataloader), desc="validating...")
            for i, (image, label) in loading:
                image, label = image.to(device), label.to(device)
                prediction = student(image)
                teacher_prediction = teacher(image)
                loss = test_criterion(prediction, label, teacher_prediction)
                
                validation_loss += loss.item()

                _, pred = torch.max(prediction, axis=1)
                validation_acc += float(torch.sum(torch.eq(pred, label)))/len(pred)
                
                loading.set_description(f"Validating[{epoch+1}/{epochs}]... Loss : {validation_loss/(i+1):.4f}, Acc : {100*validation_acc/(i+1):.2f}%")
            
            loss_hist["test"] += [validation_loss / len(val_dataloader)]
            acc_hist["test"] += [100*validation_acc/len(val_dataloader)]
            
            # Update best accuracy
            if 100*validation_acc/len(val_dataloader) > best_acc["test"]:
                best_acc["test"] = 100*validation_acc/len(val_dataloader)     
                torch.save(
                    {
                        "model_state_dict": student.state_dict()
                    }
                    , os.path.join(savedir, "best.ckpt"))
                print("Best model so far!! --> Save file")

        scheduler.step()

    results = {"best_acc" : best_acc, "loss_hist" : loss_hist, "acc_hist" : acc_hist}
    with open(os.path.join(savedir, "results.pickle"), "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(f"Best training acc : {best_acc['train']}",f"Best test acc : {best_acc['test']}")