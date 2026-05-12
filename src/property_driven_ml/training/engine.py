"""Training and evaluation engine for property-driven machine learning.

This module contains the core training and testing functions that handle
the property-driven learning loop with constraint evaluation and adversarial training.
"""

import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional

import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.training.epoch_info import EpochInfoTrain, EpochInfoTest
from property_driven_ml.training.mode import Mode


def gradnorm(loss, params):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    grads = [g.detach().norm() for g in grads if g is not None]

    if len(grads) == 0:
        return torch.tensor(0.0, device=loss.device)

    return torch.norm(torch.stack(grads))


def train(  # TODO: add task loss function as an argument
    epoch: int,
    N: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    oracle: training.Attack,
    logic: Optional[logics.Logic],
    constraint: constraints.Constraint,
    with_dl: bool,
    mode: Mode,
    alpha: float,
) -> EpochInfoTrain:
    """Train the model for one epoch with property-driven learning.

    Args:
        N: Neural network model to train.
        device: Computing device (CPU or GPU).
        train_loader: Training data loader.
        optimizer: Model optimizer.
        oracle: Attack oracle for generating adversarial examples.
        grad_norm: Gradient normalization handler.
        logic: Optional logic system for constraint evaluation.
        constraint: Constraint to enforce during training.
        with_dl: Whether to use property-driven learning.
        mode: The training mode, i.e. multi-class classification, multi-label classification, or regression.

    Returns:
        Training epoch information including metrics and sample images.
    """
    avg_pred_metric, avg_pred_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_constr_acc, avg_random_loss, avg_constr_sec, avg_constr_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_g_pred, avg_g_logic, avg_w_logic, avg_g_logic_eff, avg_grad_ratio = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    N.train()

    for _, (data, target) in enumerate(train_loader, start=1):
        x, y_target = (
            data.to(device),
            target.to(device),
        )

        # forward pass
        y = N(x)

        if mode is Mode.MultiClassClassification:
            # loss + prediction accuracy calculation
            loss = F.cross_entropy(y, y_target)
            correct = torch.mean(torch.argmax(y, dim=1).eq(y_target).float())
            avg_pred_metric += correct
        elif mode is Mode.MultiLabelClassification:
            # loss + hamming accuracy
            loss = F.binary_cross_entropy_with_logits(y, y_target)
            pred = (y > 0.0).float()  # no sigmoid to make verification easier
            correct = torch.mean((pred == y_target).float().mean(dim=1))
            avg_pred_metric += correct
        elif mode is Mode.Regression:
            # TODO: particularly ugly!
            if isinstance(constraint, constraints.AlsomitraOutputPostcondition):
                scale = 0.012
            else:
                scale = 1.0

            # loss calculation for regression
            loss = F.mse_loss(y, y_target)
            rmse = torch.sqrt(loss)
            rmse = (scale * rmse.cpu()).squeeze()
            avg_pred_metric += rmse
        else:  # TODO: can this happen?
            assert False, f"mode {mode} not supported!"  # nosec

        if with_dl:
            # TODO: adjust oracle.logic.p and logic.p according to epoch variable
            # if isinstance(logic, logics.QLL):
            #     if epoch < 20:
            #         p = 1
            #     elif epoch < 70:
            #         p = 10
            #     else:
            #         p = 50

            #     logic.p = p
            #     oracle.logic.p = p

            adv = oracle.attack(N, x, y_target, constraint)

            # forward pass for constraint accuracy (constraint satisfaction on random samples)
            with torch.no_grad():
                loss_random, sat_random = constraint.eval(
                    N, x, None, y_target, logic, reduction="mean"
                )

            # forward pass for constraint security (constraint satisfaction on adversarial samples)
            # both per-sample!
            loss_adv, sat_adv = constraint.eval(
                N, x, adv, y_target, logic, reduction=None
            )

            avg_constr_acc += sat_random
            avg_random_loss += loss_random
            avg_constr_sec += torch.mean(sat_adv)
            avg_constr_loss += torch.mean(loss_adv)

        optimizer.zero_grad(set_to_none=True)

        if not with_dl:
            loss.backward()
            optimizer.step()
        else:
            loss_pred = loss

            if isinstance(
                constraint, constraints.StrongClassificationRobustnessConstraint
            ):
                loss_logic_train = torch.mean(torch.relu(loss_adv))
            else:
                loss_logic_train = torch.mean(loss_adv)

            params = list(N.fc2.parameters())  # TODO! depends on model!!!

            g_pred = gradnorm(loss_pred, params)
            g_logic = gradnorm(loss_logic_train, params)

            w_logic = alpha * g_pred / (g_logic + 1e-8)
            w_logic = torch.clamp(w_logic, max=10.0)
            total_loss = loss_pred + w_logic.detach() * loss_logic_train

            avg_g_pred += g_pred.detach()
            avg_g_logic += g_logic.detach()
            avg_w_logic += w_logic.detach()
            avg_g_logic_eff += w_logic.detach() * g_logic.detach()
            avg_grad_ratio += (w_logic.detach() * g_logic.detach()) / (
                g_pred.detach() + 1e-8
            )

            total_loss.backward()
            optimizer.step()

        avg_pred_loss += loss

        # save one original image and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, x.size(0))

        images = dict()
        images["input"], images["adv"] = x[i], adv[i] if with_dl else None

    return EpochInfoTrain(
        pred_metric=avg_pred_metric.item() / len(train_loader),
        pred_loss=avg_pred_loss.item() / len(train_loader),
        constr_acc=(avg_constr_acc.item() / len(train_loader)) if with_dl else None,
        random_loss=(avg_random_loss.item() / len(train_loader)) if with_dl else None,
        constr_sec=(avg_constr_sec.item() / len(train_loader)) if with_dl else None,
        constr_loss=(avg_constr_loss.item() / len(train_loader)) if with_dl else None,
        pred_grad_norm=(avg_g_pred.item() / len(train_loader)) if with_dl else None,
        constr_grad_norm=(avg_g_logic.item() / len(train_loader)) if with_dl else None,
        constr_loss_weight=(avg_w_logic.item() / len(train_loader))
        if with_dl
        else None,
        weighted_constr_grad_norm=(avg_g_logic_eff.item() / len(train_loader))
        if with_dl
        else None,
        grad_ratio=(avg_grad_ratio.item() / len(train_loader)) if with_dl else None,
        input_img=images["input"],
        adv_img=images["adv"],
    )


def test(
    epoch: int,
    N: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    oracle_self: training.Attack,
    oracle_common: training.Attack,
    logic: logics.Logic,
    constraint: constraints.Constraint,
    is_baseline: bool,
    mode: Mode,
) -> EpochInfoTest:
    """Evaluate the model on test data.

    Args:
        N: Neural network model to evaluate.
        device: Computing device (CPU or GPU).
        test_loader: Test data loader.
        oracle: Attack oracle for generating adversarial examples.
        logic: Logic system for constraint evaluation.
        constraint: Constraint to evaluate.
        mode: The training mode, i.e. multi-class classification, multi-label classification, or regression.

    Returns:
        Test epoch information including metrics and sample images.
    """
    correct, constr_acc, constr_sec_self, constr_sec_common = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_pred_loss, avg_random_loss, avg_constr_loss_self, avg_constr_loss_common = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    total_samples = 0
    total_elements = 0

    N.eval()

    for _, (data, target) in enumerate(test_loader, start=1):
        x, y_target = (
            data.to(device),
            target.to(device),
        )
        total_samples += x.size(0)

        with torch.no_grad():
            # forward pass
            y = N(x)

            if mode is Mode.MultiClassClassification:
                avg_pred_loss += F.cross_entropy(y, y_target, reduction="sum")
                pred = y.max(dim=1, keepdim=True)[1]
                correct += pred.eq(y_target.view_as(pred)).sum()
            elif mode is Mode.MultiLabelClassification:
                avg_pred_loss += F.binary_cross_entropy_with_logits(
                    y, y_target, reduction="sum"
                )
                total_elements += y_target.numel()

                pred = (y > 0.0).float()
                correct += torch.sum((pred == y_target).float().mean(dim=1))
            elif mode is Mode.Regression:
                avg_pred_loss += F.mse_loss(y, y_target, reduction="sum")
            else:  # TODO: can this happen?
                assert False, f"mode {mode} not supported!"  # nosec

        # get adversarial samples (requires grad)

        # using the logic under test
        if not is_baseline:
            adv_self = oracle_self.attack(N, x, y_target, constraint)

        # using a common / reference logic
        adv_common = oracle_common.attack(N, x, y_target, constraint)

        # forward passes for constraint accuracy (constraint satisfaction on random samples) + constraint security (constraint satisfaction on adversarial samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(
                N, x, None, y_target, logic, reduction="sum"
            )

            if not is_baseline:
                loss_adv_self, sat_adv_self = constraint.eval(
                    N, x, adv_self, y_target, logic, reduction="sum"
                )
            else:
                loss_adv_self, sat_adv_self = 0, 0

            loss_adv_common, sat_adv_common = constraint.eval(
                N, x, adv_common, y_target, logic, reduction="sum"
            )

            constr_acc += sat_random
            constr_sec_self += sat_adv_self
            constr_sec_common += sat_adv_common

            avg_random_loss += loss_random
            avg_constr_loss_self += loss_adv_self
            avg_constr_loss_common += loss_adv_common

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, x.size(0))
        # Generate random sample from constraint for visualization
        random_sample = constraint.uniform_sample(x[i : i + 1], 1).squeeze(0)
        images = dict()
        images["input"], images["random"], images["adv"] = (
            x[i],
            random_sample,
            adv_common[i],
        )

    if mode in (Mode.MultiClassClassification, Mode.MultiLabelClassification):
        pred_acc = correct.item() / total_samples
    elif mode is Mode.Regression:
        # TODO: particularly ugly!
        if isinstance(constraint, constraints.AlsomitraOutputPostcondition):
            scale = 0.012
        else:
            scale = 1.0

        rmse = torch.sqrt(avg_pred_loss / total_samples)
        rmse = (scale * rmse.cpu()).item()
    else:  # TODO: can this happen?
        assert False, f"mode {mode} not supported!"  # nosec

    if mode is Mode.MultiLabelClassification:
        pred_loss = avg_pred_loss.item() / total_elements
    else:
        pred_loss = avg_pred_loss.item() / total_samples

    return EpochInfoTest(
        pred_metric=(
            pred_acc
            if mode in (Mode.MultiClassClassification, Mode.MultiLabelClassification)
            else rmse
        ),
        pred_loss=pred_loss,
        constr_acc=constr_acc.item() / total_samples,
        random_loss=avg_random_loss.item() / total_samples,
        constr_sec_self=constr_sec_self.item() / total_samples,
        constr_loss_self=avg_constr_loss_self.item() / total_samples,
        constr_sec_common=constr_sec_common.item() / total_samples,
        constr_loss_common=avg_constr_loss_common.item() / total_samples,
        input_img=images["input"],
        adv_img=images["adv"],
        random_img=images["random"],
    )
