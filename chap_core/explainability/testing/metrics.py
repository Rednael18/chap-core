import logging
from pathlib import Path

import numpy as np

from chap_core.explainability.lime import explain, explain_adaptive, perturb_vectors, produce_lime_dataset
from chap_core.models.external_model import ExternalModel
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


def eLoss(
    model,
    original_vector,
    feature_map,
    sorted_explanation,
    sampler,
    hist_df,
    fut_df,
    feature_names,
    features_hist,
    features_fut,
    horizon,
    location,
    hist_type,
    fut_type,
    feat_indices,
    y_orig,
    full_dataset=None,
    full_future_weather=None,
):

    ranked_features = [feat[0] for feat in sorted_explanation]
    num_features = len(ranked_features)

    dev_type1 = [0.0]
    dev_type2 = [0.0]

    # Threshold progression in deciles, as suggested in paper
    steps = np.linspace(0.1, 1.0, 10)

    for k_frac in steps:
        num_to_perturb = max(1, int(num_features * k_frac))

        # Type 1 perturbation
        top_k_features = ranked_features[:num_to_perturb]
        mask_type1 = np.ones(num_features)
        for idx, _ in enumerate(feature_map):
            if feature_map[idx][0] in top_k_features:
                mask_type1[idx] = 0

        # Type 2 perturbation
        bottom_k_features = ranked_features[-num_to_perturb:]  # Least important weights this time
        mask_type2 = np.ones(num_features)
        for idx, _ in enumerate(feature_map):
            if feature_map[idx][0] in bottom_k_features:
                mask_type2[idx] = 0

        pb1, pb_mask_1 = perturb_vectors(hist_df, original_vector, feat_indices, sampler, feature_map, [mask_type1])
        pb2, pb_mask_2 = perturb_vectors(hist_df, original_vector, feat_indices, sampler, feature_map, [mask_type2])

        # TODO using perturb_vectors and produce_lime_dataset directly creates a mess in the terminal from logs
        _X_type1, y_type1, _, _ = produce_lime_dataset(
            model,
            hist_df,
            fut_df,
            pb1,
            pb_mask_1,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
            full_dataset=full_dataset,
            full_future_weather=full_future_weather,
        )

        _X_type2, y_type2, _, _ = produce_lime_dataset(
            model,
            hist_df,
            fut_df,
            pb2,
            pb_mask_2,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
            full_dataset=full_dataset,
            full_future_weather=full_future_weather,
        )

        dev_type1.append(abs(y_orig - y_type1[0]))
        dev_type2.append(abs(y_orig - y_type2[0]))

    # Calculate trapizodal area
    delta_k = 1.0 / len(steps)
    auc_type1 = 0.5 * delta_k * sum(dev_type1[i - 1] + dev_type1[i] for i in range(1, len(dev_type1)))
    auc_type2 = 0.5 * delta_k * sum(dev_type2[i - 1] + dev_type2[i] for i in range(1, len(dev_type2)))

    # Delta eLoss
    delta_eloss = auc_type1 - auc_type2
    return delta_eloss, auc_type1, auc_type2


def jaccard_stability(
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    granularity: int = 10,
    num_perturbations: int = 1000,
    surrogate_name: str = "ridge",
    segmenter_name: str = "uniform",
    sampler_name: str = "background",
    weighter_name: str = "pairwise",
    seed: int | None = None,
    adaptive: bool = False,
    num_runs: int = 8,
) -> float:
    explain_fn = explain_adaptive if adaptive else explain

    explanations = []
    for i in range(num_runs):
        logger.info(f"Calculating Jaccard Stability metric, run {i}/{num_runs}")
        results = explain_fn(
            model=model,
            dataset=dataset,
            location=location,
            horizon=horizon,
            granularity=granularity,
            num_perturbations=num_perturbations,
            surrogate_name=surrogate_name,
            segmenter_name=segmenter_name,
            sampler_name=sampler_name,
            weighter_name=weighter_name,
            seed=seed,
            timed=False,
            save=False,
        )
        explanations.append(results)

    num_features = len(explanations[0])
    total_jaccard = 0.0
    count = 0
    for k in range(1, num_features + 1):
        top_k_sets = [{feat[0] for feat in exp[:k]} for exp in explanations]
        for i in range(len(top_k_sets)):
            for j in range(i + 1, len(top_k_sets)):
                intersection = len(top_k_sets[i] & top_k_sets[j])
                union = len(top_k_sets[i] | top_k_sets[j])
                total_jaccard += intersection / union if union > 0 else 1.0
                count += 1

    return total_jaccard / count if count > 0 else 0.0


if __name__ == "__main__":
    from chap_core.cli_endpoints._common import discover_geojson, load_dataset_from_csv

    dataset_csv = Path("example_data/nicaragua_weekly_data.csv")
    geojson_path = discover_geojson(dataset_csv)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path)

    template = ModelTemplate.from_directory_or_github_url(
        "runs/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6/2026-04-10_12-38-46_dcada249"
    )
    with template:
        model = template.get_model()

        stability = jaccard_stability(
            model=model,
            dataset=dataset,
            location="boaco",
            horizon=3,
            granularity=8,
            num_perturbations=50,
            surrogate_name="ridge",
            segmenter_name="uniform",
            sampler_name="fourier",
            adaptive=True,
        )
        print(f"Jaccard stability: {stability:.4f}")
