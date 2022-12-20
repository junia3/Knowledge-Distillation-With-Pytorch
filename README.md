![header](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=250&section=header&text=Knowledge%20Distillation%20with%20Pytorch&fontSize=45&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
<!-- 
<p align="center"><a href="#">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:F9D976,100:F39F86&height=250&section=header&text="Knowledge distillation" &fontSize=40&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40" alt="header" />
</a></p>
 -->

- [Train CIFAR-10 with knowledge distillation method](#train-cifar-10-with-knowledge-distillation-method)
- [Knowledge distillation loss](#knowledge-distillation-loss)
- [Step (1) Train teacher network](#step--1--train-teacher-network)
- [Pre-trained teacher network](#pre-trained-teacher-network)
- [Step (2) Train student network](#step--2--train-student-network)

## Train CIFAR-10 with knowledge distillation method
This project is held by STDL(Special Topics in Deep learning) class, Yonsei University.   
There are some requirements. Pytorch installation is optional with your own computer.
You can check on [my blog](https://junia3.github.io/blog/transfer) post to read about various learning methods in deep learning. It includes knowledge distillation. 

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torchsummary
pip install tqdm
pip install pickle5
```

## Knowledge distillation loss
$$ 
\begin{aligned}
  \mathcal{L} =& T^2 \lambda \mathcal{L_t} + (1-\lambda) \mathcal{L_s} \newline
  \mathcal{L_s} = \mathcal{L}_{CE}(Y, S(X, 1)),&~\mathcal{L_t} = \mathcal{L}_{CE}(S(X, T),~R(X, T)) \newline
  q_i =& \frac{exp(z_i/T)}{\sum_j exp(z_i/T)} \newline
\end{aligned}
$$

Where student network $S$ and teacher network $R$ exists. The pytorch code is implemented as below

```python3
class KDLoss(nn.Module):
    def __init__(self, T, alpha):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_pred, target, teacher_pred):
        student_loss = nn.CrossEntropyLoss()(student_pred, target)
        soft_target = F.softmax(teacher_pred/self.T, dim=1) # softmax on teacher prediction
        soft_pred = F.log_softmax(student_pred/self.T, dim=1) # log softmax on student prediction
        teacher_loss = nn.KLDivLoss(reduction="batchmean")(soft_pred, soft_target) # KL divergence -> cross entropy loss
        loss = self.alpha*self.T*self.T*teacher_loss + (1-self.alpha)*student_loss
        return loss
```

## Step (1) Train teacher network
Before train the student network, you should make pre-trained teacher network for CIFAR-10 dataset.

```bash
python train.py --mode teacher
```

And also you can change any code in train.py if you want to use another optimizer or scheduler.

|Option|Configuration|
|:---:|:---:|
|Epochs|100|
|Learning rate|scratch : 0.01<br>finetune : 0.0001|
|Optimizer|Adam|
|Scheduler|CosineAnnealingLR|

---

## Pre-trained teacher network(Modified structure of ResNet-18)
Or you can just download my model below.
|Model|Training Accuracy|Test Accuracy|
|:---:|:---:|:---:|
|Teacher network([download](https://drive.google.com/file/d/1av6cD6rdsSQ83ojM9k5Woc7lfn153IZR/view?usp=share_link))|100%|95.39%|

After download above file, you must locate pre-trained file to "teacher/best.ckpt" before train student model with knowledge distillation loss.

## Step (2) Train student network
After create the pre-trained teacher network, you can train student network with KD loss.

```bash
python train.py --mode student
```
You can modify $\alpha$ and temperature $T$ values.

## Pre-trained student network
I simply train student model in model.py. If you want to train another structure, build your own student and teacher relationships.
|Model|Training Accuracy|Test Accuracy|
|:---:|:---:|:---:|
|Teacher network([download](https://drive.google.com/file/d/1av6cD6rdsSQ83ojM9k5Woc7lfn153IZR/view?usp=share_link))|100%|95.39%|
|Student network([download](https://drive.google.com/file/d/1H4CuX07hNbh146CMfXrulpMgFrKpzecs/view?usp=share_link))|85.14%|84.84|

Training studnet network with KD loss. $T = 20$ and $\lambda = 0.1$.


![footer](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=150&section=footer&animation=fadeIn&fontColor=FFFFFF&fontAlignY=40)
