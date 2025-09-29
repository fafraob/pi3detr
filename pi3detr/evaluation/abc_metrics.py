import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.functional import average_precision
from scipy.spatial import KDTree


def calc_chamfer_distance(pred_points, gt_points):
    """Calculate chamfer distance and bidirectional hausdorff distance."""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf"), float("inf")

    tree_pred = KDTree(pred_points)
    tree_gt = KDTree(gt_points)

    dist_pred2gt, _ = tree_gt.query(pred_points)
    dist_gt2pred, _ = tree_pred.query(gt_points)

    chamfer_dist = np.mean(dist_pred2gt**2) + np.mean(dist_gt2pred**2)
    bhaussdorf_dist = (dist_pred2gt.max() + dist_gt2pred.max()) / 2

    return chamfer_dist, bhaussdorf_dist


class ChamferMAP(Metric):
    def __init__(self, chamfer_thresh=0.05, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.chamfer_thresh = chamfer_thresh
        self.class_names = {
            1: "mAP_bspline",
            2: "mAP_line",
            3: "mAP_circle",
            4: "mAP_arc",
        }

        self.add_state("all_scores", default=[], dist_reduce_fx="cat")
        self.add_state("all_matches", default=[], dist_reduce_fx="cat")
        self.add_state("all_classes", default=[], dist_reduce_fx="cat")

    def pairwise_chamfer_distance_batch(self, pred, gt):
        """
        pred: [P, 64, 3]
        gt:   [G, 64, 3]
        returns: [P, G] chamfer distances
        """
        P, G = pred.size(0), gt.size(0)

        # Reshape for pairwise comparison
        pred_exp = pred.unsqueeze(1)  # [P, 1, 64, 3]
        gt_exp = gt.unsqueeze(0)  # [1, G, 64, 3]

        # Compute pairwise distances between points
        dists = torch.cdist(pred_exp, gt_exp, p=2)  # [P, G, 64, 64]

        a2b = dists.min(dim=3).values.mean(dim=2)  # [P, G]
        b2a = dists.min(dim=2).values.mean(dim=2)  # [P, G]

        return a2b + b2a  # [P, G]

    def update(self, outputs, batch):
        B = outputs["pred_class"].shape[0]
        y_curves = batch.y_curve_64  # [total_gt, 64, 3]
        y_cls = batch.y_cls  # [total_gt]
        num_curves_per_batch = batch.num_curves.tolist()

        gt_splits = torch.split(y_curves, num_curves_per_batch, dim=0)
        cls_splits = torch.split(y_cls, num_curves_per_batch, dim=0)

        pred_classes_all = outputs["pred_class"].softmax(dim=-1)  # [B, N, C]
        for b in range(B):
            pred_classes = pred_classes_all[b]  # [N, C]

            preds_all = {
                1: outputs["pred_bspline_points"][b],  # [N, 64, 3]
                2: outputs["pred_line_points"][b],
                3: outputs["pred_circle_points"][b],
                4: outputs["pred_arc_points"][b],
            }

            for cls in self.class_names.keys():
                pred_points = preds_all[cls]  # [P, 64, 3]
                scores = pred_classes[:, cls]  # [P]

                gt_points = gt_splits[b][cls_splits[b] == cls]  # [G, 64, 3]
                if gt_points.size(0) == 0:
                    self.all_scores.append(scores)
                    self.all_matches.append(torch.zeros_like(scores))
                    self.all_classes.append(torch.full_like(scores, cls))
                    continue

                chamfer = self.pairwise_chamfer_distance_batch(pred_points, gt_points)
                used_gt = torch.zeros(
                    gt_points.size(0), dtype=torch.bool, device=pred_points.device
                )
                matches = torch.zeros(pred_points.size(0), device=pred_points.device)

                for i in range(pred_points.size(0)):
                    dists = chamfer[i]
                    min_dist, min_idx = dists.min(0)
                    if min_dist < self.chamfer_thresh and not used_gt[min_idx]:
                        matches[i] = 1.0
                        used_gt[min_idx] = True

                self.all_scores.append(scores)
                self.all_matches.append(matches)
                self.all_classes.append(torch.full_like(matches, cls))

    def compute(self):
        if not self.all_scores:
            return {cls_name: 0.0 for cls_name in self.class_names.values()}

        scores = torch.cat(self.all_scores)
        matches = torch.cat(self.all_matches)
        classes = torch.cat(self.all_classes)

        result = {}
        ap_values = []

        for cls in self.class_names.keys():
            mask = classes == cls
            if mask.sum() == 0 or torch.sum(matches[mask]) == 0:
                ap = torch.tensor(0.0, device=self.device)
            else:
                ap = average_precision(
                    scores[mask], matches[mask].to(torch.int32), task="binary"
                )
            result[self.class_names[cls]] = ap.item()
            ap_values.append(ap)

        result["mAP"] = torch.stack(ap_values).mean().item()
        return result


class ChamferIntervalMetric(Metric):

    def __init__(self, interval=0.01, map_cd_thresh=0.005, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.interval = interval
        self.add_state("total_cd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_cd_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_bhd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_bhd_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("valid_count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.map_cd_thresh = map_cd_thresh
        self.map_cls_names = {
            1: "mAP_bspline",
            2: "mAP_line",
            3: "mAP_circle",
            4: "mAP_arc",
        }

        self.add_state("map_all_scores", default=[], dist_reduce_fx="cat")
        self.add_state("map_all_matches", default=[], dist_reduce_fx="cat")
        self.add_state("map_all_classes", default=[], dist_reduce_fx="cat")

    def sample_curve_by_interval(self, points, interval, force_last=False):
        """Sample points along the curve at fixed length `interval`."""
        if len(points) < 2:
            return points

        edges = np.array([[j, j + 1] for j in range(len(points) - 1)])
        edge_lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)

        samples = [points[0]]
        distance_accum = 0.0
        next_sample_dist = interval
        edge_index = 0

        while edge_index < len(edges):
            p0 = points[edges[edge_index, 0]]
            p1 = points[edges[edge_index, 1]]
            edge_vec = p1 - p0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                edge_index += 1
                continue

            while distance_accum + edge_len >= next_sample_dist:
                t = (next_sample_dist - distance_accum) / edge_len
                sample = p0 + t * edge_vec
                samples.append(sample)
                next_sample_dist += interval

            distance_accum += edge_len
            edge_index += 1

        if force_last and not np.allclose(samples[-1], points[-1]):
            samples.append(points[-1])

        return np.array(samples)

    def update(self, data, batch):

        # Get ground truth curves
        y_curves = batch.y_curve_64.cpu().numpy()  # [total_gt, 64, 3]
        num_curves_per_batch = batch.num_curves.tolist()

        # Since batch size is 1 in your case
        B = 1

        # Sample ground truth curves
        gt_points_list = []
        gt_cls_list = []
        for i, gt_curve in enumerate(y_curves):
            if len(gt_curve) < 2 or np.any(np.isnan(gt_curve)):
                continue
            sampled_gt = self.sample_curve_by_interval(
                gt_curve, self.interval, force_last=True
            )
            if len(sampled_gt) > 0 and np.all(np.isfinite(sampled_gt)):
                gt_points_list.append(sampled_gt)
                gt_cls_list.append(batch.y_cls[i].cpu().item())

        # Sample predicted curves from post-processed data
        pred_points_list = []
        pred_cls_list = []
        pred_score_list = []
        for polyline, cls in zip(
            data.polylines.cpu().numpy(), data.polyline_class.cpu().numpy()
        ):
            if cls == 0:  # Skip background class
                continue

            if len(polyline) < 2 or np.any(np.isnan(polyline)):
                continue

            sampled_pred = self.sample_curve_by_interval(
                polyline, self.interval, force_last=True
            )
            if len(sampled_pred) > 0 and np.all(np.isfinite(sampled_pred)):
                pred_points_list.append(sampled_pred)
                pred_cls_list.append(int(cls))
                pred_score_list.append(data.polyline_score[i].cpu().item())

        if len(gt_points_list) == 0 and len(pred_points_list) == 0:
            # No ground truth and no predictions, no penalty
            self.count += 1
            return
        elif len(gt_points_list) == 0:
            # Penalize no ground truth
            self.count += 1
            scores = torch.tensor(pred_score_list)
            self.map_all_scores.append(scores)
            self.map_all_matches.append(torch.zeros_like(scores))
            self.map_all_classes.append(torch.tensor(pred_cls_list))
            return
        elif len(pred_points_list) == 0:
            # Penalize no predictions
            self.count += 1
            cls_list = torch.tensor(gt_cls_list, dtype=torch.float32)
            self.map_all_scores.append(torch.zeros_like(cls_list))
            self.map_all_matches.append(torch.ones_like(cls_list))
            self.map_all_classes.append(torch.tensor(cls_list))
            return

        # calculate mAP
        for cls in self.map_cls_names.keys():
            mask = torch.tensor(pred_cls_list) == cls
            pred_curves = [curve for i, curve in enumerate(pred_points_list) if mask[i]]
            pred_scores = torch.tensor(pred_score_list)[mask]
            gt_curves = [
                curve for i, curve in enumerate(gt_points_list) if cls == gt_cls_list[i]
            ]
            if len(pred_curves) == 0 and len(gt_curves) != 0:
                scores = torch.zeros(len(gt_curves))
                self.map_all_scores.append(scores)
                self.map_all_matches.append(torch.zeros_like(scores))
                self.map_all_classes.append(torch.full_like(scores, cls))
                continue
            if len(gt_curves) == 0:
                self.map_all_scores.append(pred_scores)
                self.map_all_matches.append(torch.zeros_like(pred_scores))
                self.map_all_classes.append(torch.full_like(pred_scores, cls))
                continue

            # get [P, G] matrix of chamfer distances
            cd_matrix = torch.ones((len(pred_curves), len(gt_curves))) * float("inf")
            for i, pred_curve in enumerate(pred_curves):
                for j, gt_curve in enumerate(gt_curves):
                    cd_matrix[i, j] = calc_chamfer_distance(pred_curve, gt_curve)[0]

            used_gt = set()
            matches = torch.zeros(len(pred_curves))
            for i in range(len(pred_curves)):
                dists = cd_matrix[i]
                min_dist, min_idx = dists.min(0)
                if min_dist < self.map_cd_thresh and min_idx not in used_gt:
                    matches[i] = 1.0
                    used_gt.add(min_idx)

            self.map_all_scores.append(pred_scores)
            self.map_all_matches.append(matches)
            self.map_all_classes.append(torch.full_like(pred_scores, cls))

        pred_points = np.concatenate(pred_points_list, axis=0)
        gt_points = np.concatenate(gt_points_list, axis=0)
        # Calculate distances
        cd, bhd = calc_chamfer_distance(pred_points, gt_points)

        self.total_cd += torch.tensor(cd)
        self.total_cd_sq += torch.tensor(cd**2)
        self.total_bhd += torch.tensor(bhd)
        self.total_bhd_sq += torch.tensor(bhd**2)
        self.count += 1
        self.valid_count += 1 if len(pred_points) > 0 else 0

    def compute(self):
        if not self.map_all_scores:
            return {cls_name: 0.0 for cls_name in self.map_cls_names.values()}
        scores = torch.cat(self.map_all_scores)
        matches = torch.cat(self.map_all_matches)
        classes = torch.cat(self.map_all_classes)
        map_result = {}
        ap_values = []

        for cls in self.map_cls_names.keys():
            mask = classes == cls
            if mask.sum() == 0 or torch.sum(matches[mask]) == 0:
                ap = torch.tensor(0.0, device=self.device)
            else:
                ap = average_precision(
                    scores[mask], matches[mask].to(torch.int32), task="binary"
                )
            map_result[self.map_cls_names[cls]] = ap.item()
            ap_values.append(ap)

        map_result["mAP"] = torch.stack(ap_values).mean().item()

        if self.count == 0:
            return {"chamfer_distance": 0.0, "bidirectional_hausdorff": 0.0}

        mean_cd = (self.total_cd / self.valid_count).item()
        mean_bhd = (self.total_bhd / self.valid_count).item()
        results = {
            "chamfer_distance": mean_cd,
            "chamfer_distance_std": (self.total_cd_sq / self.valid_count - (mean_cd**2))
            .sqrt()
            .item(),
            "bidirectional_hausdorff": mean_bhd,
            "bidirectional_hausdorff_std": (
                self.total_bhd_sq / self.valid_count - (mean_bhd**2)
            )
            .sqrt()
            .item(),
        }
        results.update(map_result)
        return results
