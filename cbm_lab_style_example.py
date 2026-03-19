"""
CBM Project rewritten in the Lab's coding style.

In the labs, every model class owns its optimizer, criterion, training loop,
and evaluation method. This file shows what your project would look like
if you followed that same pattern.

Compare with your current approach where train_baseline(), train_concept_predictor(),
etc. are standalone functions that receive the model as an argument.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, f1_score


# ---------------------------------------------------------------------------
# Configuration (same as your notebook)
# ---------------------------------------------------------------------------
CONCEPT_NAMES = [
    "Mouth_Slightly_Open", "High_Cheekbones", "Chubby",
    "Narrow_Eyes", "Bags_Under_Eyes", "Big_Lips",
    "Big_Nose", "Pointy_Nose", "Bushy_Eyebrows", "Arched_Eyebrows",
]
NUM_CONCEPTS = len(CONCEPT_NAMES)
# These would be set after loading the dataset:
# SMILING_IDX = dataset.attr_names.index('Smiling')
# CONCEPT_IDXS = [list(dataset.attr_names).index(n) for n in CONCEPT_NAMES]


# =========================================================================
# 1) BASELINE CLASSIFIER  (x -> y)
#    Lab style: the class owns everything
# =========================================================================
class BaselineClassifier_LabStyle(nn.Module):
    """
    Compare with the lab's Lenet5_extended_GPU class.

    In the lab, __init__ creates the architecture AND the optimizer/criterion.
    trainloop() is a method on the model, not a standalone function.
    eval_performance() is also a method.
    """

    def __init__(self, smiling_idx, epochs=5, lr=1e-3):
        super().__init__()

        # --- Architecture (like Lenet5.__init__) ---
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(512, 1)

        # --- Training config (like Lenet5_extended_GPU.__init__) ---
        self.smiling_idx = smiling_idx
        self.lr = lr
        self.epochs = epochs
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        # --- Loss tracking (identical to the lab pattern) ---
        self.loss_during_training = []
        self.valid_loss_during_training = []

        # --- Device handling (identical to lab) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)  # (B, 512)
        return self.classifier(features)                    # (B, 1)

    def trainloop(self, trainloader, validloader):
        """
        Mirrors the lab's trainloop method exactly in structure.

        Lab pattern:
            for e in range(self.epochs):
                running_loss = 0.
                for images, labels in trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optim.zero_grad()
                    out = self.forward(images)
                    loss = self.criterion(out, labels)
                    running_loss += loss.item()
                    loss.backward()
                    self.optim.step()
                self.loss_during_training.append(running_loss / len(trainloader))
                # then validation block ...

        Your current project does the same thing, but as a standalone function
        that receives the model as an argument. The logic is identical.
        """
        best_val_auroc = 0.0
        best_weights = None

        for e in range(self.epochs):
            start_time = time.time()

            # ---- Training phase (same as lab) ----
            self.train()
            running_loss = 0.

            for images, labels in trainloader:
                images = images.to(self.device)
                targets = labels[:, self.smiling_idx].float().to(self.device)

                self.optim.zero_grad()
                logits = self.forward(images).squeeze(1)
                loss = self.criterion(logits, targets)
                running_loss += loss.item()
                loss.backward()
                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # ---- Validation phase (same as lab) ----
            with torch.no_grad():
                self.eval()
                val_running_loss = 0.
                all_labels, all_probs = [], []

                for images, labels in validloader:
                    images = images.to(self.device)
                    targets = labels[:, self.smiling_idx].float()

                    logits = self.forward(images).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()

                    val_loss = self.criterion(
                        logits.cpu(),
                        targets
                    )
                    val_running_loss += val_loss.item()

                    all_labels.append(targets.numpy())
                    all_probs.append(probs)

                self.valid_loss_during_training.append(
                    val_running_loss / len(validloader)
                )

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            val_auroc = roc_auc_score(all_labels, all_probs)

            elapsed = time.time() - start_time
            print(
                f"Epoch {e+1}/{self.epochs} | "
                f"Loss: {self.loss_during_training[-1]:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_weights = {k: v.clone() for k, v in self.state_dict().items()}
                print(f"  --> New best model (AUROC: {best_val_auroc:.4f})")

        self.load_state_dict(best_weights)
        print(f"\nTraining complete. Best Val AUROC: {best_val_auroc:.4f}")

    def eval_performance(self, dataloader):
        """
        Mirrors the lab's eval_performance method.

        In the lab this returns a single accuracy number.
        Here we return (accuracy, auroc) because the project requires both.
        """
        self.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                targets = labels[:, self.smiling_idx].float()

                logits = self.forward(images).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_labels.append(targets.numpy())
                all_probs.append(probs)

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        preds = (all_probs >= 0.5).astype(int)

        accuracy = (preds == all_labels).mean()
        auroc = roc_auc_score(all_labels, all_probs)
        return accuracy, auroc


# =========================================================================
# 2) CONCEPT PREDICTOR  (x -> c)
#    Lab style
# =========================================================================
class ConceptPredictor_LabStyle(nn.Module):
    """
    Multi-label binary predictor for 10 concept attributes.

    In lab style, the class owns pos_weight, criterion, optimizer, and
    the training loop. In your current project, pos_weight is computed
    externally and passed to a standalone train_concept_predictor() function.
    """

    def __init__(self, concept_idxs, pos_weight, epochs=5, lr=1e-3):
        super().__init__()

        # --- Architecture ---
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Linear(512, NUM_CONCEPTS)

        # --- Training config ---
        self.concept_idxs = concept_idxs
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optim = optim.Adam(self.parameters(), self.lr)

        # --- Loss tracking ---
        self.loss_during_training = []
        self.valid_loss_during_training = []

        # --- Device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = self.criterion.to(self.device)
        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        return self.head(features)  # (B, 10)

    def trainloop(self, trainloader, validloader):
        best_val_f1 = 0.0
        best_weights = None

        for e in range(self.epochs):
            start_time = time.time()

            # ---- Training ----
            self.train()
            running_loss = 0.

            for images, labels in trainloader:
                images = images.to(self.device)
                targets = labels[:, self.concept_idxs].float().to(self.device)

                self.optim.zero_grad()
                logits = self.forward(images)
                loss = self.criterion(logits, targets)
                running_loss += loss.item()
                loss.backward()
                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # ---- Validation ----
            with torch.no_grad():
                self.eval()
                all_labels, all_preds = [], []

                for images, labels in validloader:
                    images = images.to(self.device)
                    targets = labels[:, self.concept_idxs].float()
                    logits = self.forward(images).cpu()
                    preds = (torch.sigmoid(logits) >= 0.5).int()
                    all_labels.append(targets.numpy())
                    all_preds.append(preds.numpy())

            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            elapsed = time.time() - start_time
            print(
                f"Epoch {e+1}/{self.epochs} | "
                f"Loss: {self.loss_during_training[-1]:.4f} | "
                f"Val Macro-F1: {macro_f1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                best_weights = {k: v.clone() for k, v in self.state_dict().items()}
                print(f"  --> New best model (Macro-F1: {best_val_f1:.4f})")

        self.load_state_dict(best_weights)
        print(f"\nTraining complete. Best Val Macro-F1: {best_val_f1:.4f}")

    def eval_performance(self, dataloader):
        self.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                targets = labels[:, self.concept_idxs].float()
                logits = self.forward(images).cpu()
                preds = (torch.sigmoid(logits) >= 0.5).int()
                all_labels.append(targets.numpy())
                all_preds.append(preds.numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        accs, f1s = [], []
        print(f"{'Concept':<25} {'Accuracy':>10} {'F1':>10}")
        print("-" * 47)
        for i, name in enumerate(CONCEPT_NAMES):
            acc = (all_preds[:, i] == all_labels[:, i]).mean()
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            accs.append(acc)
            f1s.append(f1)
            print(f"{name:<25} {acc:>10.4f} {f1:>10.4f}")
        print("-" * 47)
        print(f"{'Macro Average':<25} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f}")
        return np.mean(accs), np.mean(f1s)


# =========================================================================
# 3) CONCEPT BOTTLENECK MODEL  (x -> c -> y)  Independent training
#    Lab style
#
#    This is where the lab style gets awkward. In the lab, training is
#    always a single loop for a single model. But the independent CBM
#    requires a TWO-PHASE training process:
#      Phase 1: train x->c (the ConceptPredictor above)
#      Phase 2: freeze x->c, train c->y
#
#    The lab style forces you to handle this with inheritance, which is
#    clunkier than your current approach of just copying weights.
# =========================================================================
class ConceptBottleneckModel_LabStyle(nn.Module):
    """
    Independent CBM: inherits a pretrained concept predictor's weights,
    freezes them, and only trains the label head.

    Notice how __init__ takes a pretrained ConceptPredictor as an argument
    and copies its weights. In the lab style, this is like how the
    Tran_Eval class in Lab Part 2 takes a pretrained DenseNet model
    as an argument:

        class Tran_Eval():
            def __init__(self, model, maxiter=500, lr=0.001):
                self.model = model
                ...

    The lab's DenseNet transfer learning example is actually the closest
    ancestor to this pattern.
    """

    def __init__(self, pretrained_concept_model, smiling_idx, concept_idxs,
                 epochs=5, lr=1e-3):
        super().__init__()

        # --- Architecture: copy from pretrained concept predictor ---
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.concept_head = nn.Linear(512, NUM_CONCEPTS)
        self.label_head = nn.Linear(NUM_CONCEPTS, 1)

        # Copy pretrained weights (this has no lab equivalent)
        self.backbone.load_state_dict(pretrained_concept_model.backbone.state_dict())
        self.concept_head.load_state_dict(pretrained_concept_model.head.state_dict())

        # Freeze backbone and concept head (lab does this for DenseNet)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.concept_head.parameters():
            param.requires_grad = False

        # --- Training config ---
        # Note: optimizer only sees trainable params (label_head)
        # In the lab's DenseNet example:
        #   self.optim = optim.Adam(self.model.classifier.parameters(), self.lr)
        # Same idea here.
        self.smiling_idx = smiling_idx
        self.concept_idxs = concept_idxs
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.lr
        )

        self.loss_during_training = []
        self.valid_loss_during_training = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)
        label_logit = self.label_head(concept_probs)
        return concept_logits, label_logit

    def trainloop(self, trainloader, validloader):
        """
        Only trains the label_head (11 parameters).
        Backbone and concept_head are frozen.
        """
        best_val_auroc = 0.0
        best_weights = None

        for e in range(self.epochs):
            start_time = time.time()
            self.train()
            running_loss = 0.

            for images, labels in trainloader:
                images = images.to(self.device)
                y_target = labels[:, self.smiling_idx].float().to(self.device)

                self.optim.zero_grad()
                _, label_logit = self.forward(images)
                loss = self.criterion(label_logit.squeeze(1), y_target)
                running_loss += loss.item()
                loss.backward()
                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation
            with torch.no_grad():
                self.eval()
                all_labels, all_probs = [], []

                for images, labels in validloader:
                    images = images.to(self.device)
                    _, label_logit = self.forward(images)
                    probs = torch.sigmoid(label_logit.squeeze(1)).cpu().numpy()
                    all_labels.append(labels[:, self.smiling_idx].numpy())
                    all_probs.append(probs)

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            val_auroc = roc_auc_score(all_labels, all_probs)

            elapsed = time.time() - start_time
            print(
                f"Epoch {e+1}/{self.epochs} | "
                f"Loss: {self.loss_during_training[-1]:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_weights = {k: v.clone() for k, v in self.state_dict().items()}
                print(f"  --> New best (AUROC: {best_val_auroc:.4f})")

        self.load_state_dict(best_weights)
        print(f"\nTraining complete. Best Val AUROC: {best_val_auroc:.4f}")

    def eval_performance(self, dataloader):
        self.eval()
        all_y_labels, all_y_probs = [], []
        all_c_labels, all_c_preds = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                c_targets = labels[:, self.concept_idxs].float()
                y_targets = labels[:, self.smiling_idx].float()

                concept_logits, label_logit = self.forward(images)
                y_probs = torch.sigmoid(label_logit.squeeze(1)).cpu().numpy()
                c_preds = (torch.sigmoid(concept_logits) >= 0.5).int().cpu().numpy()

                all_y_labels.append(y_targets.numpy())
                all_y_probs.append(y_probs)
                all_c_labels.append(c_targets.numpy())
                all_c_preds.append(c_preds)

        all_y_labels = np.concatenate(all_y_labels)
        all_y_probs = np.concatenate(all_y_probs)
        all_c_labels = np.concatenate(all_c_labels)
        all_c_preds = np.concatenate(all_c_preds)

        y_preds = (all_y_probs >= 0.5).astype(int)
        accuracy = (y_preds == all_y_labels).mean()
        auroc = roc_auc_score(all_y_labels, all_y_probs)

        print(f"  Smiling Accuracy: {accuracy:.4f}")
        print(f"  Smiling AUROC:    {auroc:.4f}")

        print(f"\n  {'Concept':<25} {'Accuracy':>10} {'F1':>10}")
        print(f"  {'-'*47}")
        accs, f1s = [], []
        for i, name in enumerate(CONCEPT_NAMES):
            acc = (all_c_preds[:, i] == all_c_labels[:, i]).mean()
            f1 = f1_score(all_c_labels[:, i], all_c_preds[:, i], zero_division=0)
            accs.append(acc)
            f1s.append(f1)
            print(f"  {name:<25} {acc:>10.4f} {f1:>10.4f}")
        print(f"  {'-'*47}")
        print(f"  {'Macro Average':<25} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f}")

        return accuracy, auroc


# =========================================================================
# 4) HYBRID CBM with side channel  (lab style)
#    Lab style with dropout sweep
# =========================================================================
class HybridCBM_LabStyle(nn.Module):
    """
    y = f(c) + s(x)

    This is where the lab style starts to feel forced. The dropout
    probability is baked into the class at construction time, just like
    Lenet5_Drop in Lab Part 1 takes prob as a constructor argument.

    But running a sweep means you create 6 separate class instances,
    each with its own optimizer and loss tracking. In your current
    project style, you create the model and pass it to a standalone
    function, which is cleaner for sweeps.
    """

    def __init__(self, concept_idxs, smiling_idx, pos_weight,
                 side_dropout=0.0, epochs=5, lr=1e-4, concept_loss_weight=0.5):
        super().__init__()

        # --- Architecture ---
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.concept_head = nn.Linear(512, NUM_CONCEPTS)
        self.label_head = nn.Linear(NUM_CONCEPTS, 1)

        # Side channel with dropout (like Lenet5_Drop's self.dropout)
        self.side_channel = nn.Sequential(
            nn.Dropout(p=side_dropout),
            nn.Linear(512, 1)
        )
        self.side_dropout_p = side_dropout

        # --- Training config ---
        self.concept_idxs = concept_idxs
        self.smiling_idx = smiling_idx
        self.concept_loss_weight = concept_loss_weight
        self.lr = lr
        self.epochs = epochs

        self.label_criterion = nn.BCEWithLogitsLoss()
        self.concept_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.loss_during_training = []
        self.valid_loss_during_training = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concept_criterion = self.concept_criterion.to(self.device)
        self.to(self.device)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)
        label_logit_concept = self.label_head(concept_probs)
        label_logit_side = self.side_channel(features)
        label_logit = label_logit_concept + label_logit_side
        return concept_logits, label_logit, label_logit_concept, label_logit_side

    def trainloop(self, trainloader, validloader):
        best_val_auroc = 0.0
        best_weights = None

        for e in range(self.epochs):
            start_time = time.time()
            self.train()
            running_loss = 0.

            for images, labels in trainloader:
                images = images.to(self.device)
                y_target = labels[:, self.smiling_idx].float().to(self.device)
                c_targets = labels[:, self.concept_idxs].float().to(self.device)

                self.optim.zero_grad()
                concept_logits, label_logit, _, _ = self.forward(images)

                loss_label = self.label_criterion(label_logit.squeeze(1), y_target)
                loss_concept = self.concept_criterion(concept_logits, c_targets)
                loss = loss_label + self.concept_loss_weight * loss_concept

                running_loss += loss.item()
                loss.backward()
                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Validation
            with torch.no_grad():
                self.eval()
                all_labels, all_probs = [], []

                for images, labels in validloader:
                    images = images.to(self.device)
                    _, label_logit, _, _ = self.forward(images)
                    probs = torch.sigmoid(label_logit.squeeze(1)).cpu().numpy()
                    all_labels.append(labels[:, self.smiling_idx].numpy())
                    all_probs.append(probs)

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            val_auroc = roc_auc_score(all_labels, all_probs)

            elapsed = time.time() - start_time
            print(
                f"[p={self.side_dropout_p}] Epoch {e+1}/{self.epochs} | "
                f"Loss: {self.loss_during_training[-1]:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_weights = {k: v.clone() for k, v in self.state_dict().items()}
                print(f"  --> New best (AUROC: {best_val_auroc:.4f})")

        self.load_state_dict(best_weights)
        print(f"\nTraining complete. Best Val AUROC: {best_val_auroc:.4f}")

    def eval_performance(self, dataloader, verbose=True):
        self.eval()
        all_y_labels, all_y_probs = [], []
        all_c_labels, all_c_preds = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                c_targets = labels[:, self.concept_idxs].float()
                y_targets = labels[:, self.smiling_idx].float()

                concept_logits, label_logit, _, _ = self.forward(images)
                y_probs = torch.sigmoid(label_logit.squeeze(1)).cpu().numpy()
                c_preds = (torch.sigmoid(concept_logits) >= 0.5).int().cpu().numpy()

                all_y_labels.append(y_targets.numpy())
                all_y_probs.append(y_probs)
                all_c_labels.append(c_targets.numpy())
                all_c_preds.append(c_preds)

        all_y_labels = np.concatenate(all_y_labels)
        all_y_probs = np.concatenate(all_y_probs)
        all_c_labels = np.concatenate(all_c_labels)
        all_c_preds = np.concatenate(all_c_preds)

        y_preds = (all_y_probs >= 0.5).astype(int)
        accuracy = (y_preds == all_y_labels).mean()
        auroc = roc_auc_score(all_y_labels, all_y_probs)
        macro_f1 = f1_score(all_c_labels, all_c_preds, average="macro", zero_division=0)

        if verbose:
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUROC:    {auroc:.4f}")
            print(f"  Concept Macro-F1: {macro_f1:.4f}")

        return accuracy, auroc, macro_f1


# =========================================================================
# USAGE EXAMPLE (what the notebook cells would look like)
#
# This is the lab-style calling convention: instantiate, call trainloop,
# call eval_performance -- all as methods on the object.
# =========================================================================

"""
# --- Baseline ---
my_baseline = BaselineClassifier_LabStyle(smiling_idx=SMILING_IDX, epochs=5, lr=1e-3)
my_baseline.trainloop(trainloader, validloader)
baseline_acc, baseline_auroc = my_baseline.eval_performance(testloader)

# --- Concept Predictor ---
my_concepts = ConceptPredictor_LabStyle(
    concept_idxs=CONCEPT_IDXS, pos_weight=pos_weight, epochs=5, lr=1e-3
)
my_concepts.trainloop(trainloader, validloader)
my_concepts.eval_performance(testloader)

# --- Independent CBM ---
my_cbm = ConceptBottleneckModel_LabStyle(
    pretrained_concept_model=my_concepts,
    smiling_idx=SMILING_IDX,
    concept_idxs=CONCEPT_IDXS,
    epochs=5, lr=1e-3
)
my_cbm.trainloop(trainloader, validloader)
cbm_acc, cbm_auroc = my_cbm.eval_performance(testloader)

# --- Hybrid Dropout Sweep ---
# In lab style, you create a new object for each dropout value.
# Compare with Lab Part 1 Cell 30:
#   my_CNN_GPU_Drop = Lenet5_extended_GPU_Drop(dimx=32, nlabels=10, prob=0.5, epochs=20)
#   my_CNN_GPU_Drop.trainloop(trainloader, validloader)

sweep_results = {}
for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    my_hybrid = HybridCBM_LabStyle(
        concept_idxs=CONCEPT_IDXS,
        smiling_idx=SMILING_IDX,
        pos_weight=pos_weight,
        side_dropout=p, epochs=5, lr=1e-4
    )
    my_hybrid.trainloop(trainloader, validloader)
    acc, auroc, cf1 = my_hybrid.eval_performance(testloader)
    sweep_results[p] = {"accuracy": acc, "auroc": auroc, "concept_f1": cf1}

# Then plot sweep_results as before...
"""
