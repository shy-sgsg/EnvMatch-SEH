"""Phase I-5 geographical forests, spatial SHAP/ALE, and SVC-GAM exploration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_spatial_ml_mplconfig")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import yaml
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

import phase_i_spatial as spatial

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
ENV = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
KEY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
LABELS = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM", "CEC": "CEC", "pH": "pH",
          "SW": "SW", "BD": "BD", "MBC": "MBC", "RESP": "RESP"}
SOIL_COLORS = {"26": "#277DA1", "27": "#E07A5F", "42": "#70A288", "85": "#9B5DE5", "other": "#7A8793"}
try:
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except ValueError:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _read_pixels(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, Path]:
    db = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path, health_path = db / "environment_erosion.parquet", db / "health.parquet"
    ecols = ["cell_id", "row", "col", "x", "y", "Aa_20yr", "SoilType", *config["environment_covariates"]]
    env = pd.read_parquet(env_path, columns=ecols)
    health = pd.read_parquet(health_path, columns=["cell_id", *config["outcomes"]])
    data = env.merge(health, on="cell_id", how="inner", validate="one_to_one")
    if len(data) != len(env) or not data["cell_id"].is_unique:
        raise ValueError("Environment and health databases must align one-to-one by cell_id.")
    data["log1p_Aa"] = np.log1p(pd.to_numeric(data["Aa_20yr"], errors="coerce"))
    return data, env_path, health_path


def _aggregate_pixels(frame: pd.DataFrame, outcome: str, config: dict[str, Any]) -> pd.DataFrame:
    env = list(config["environment_covariates"])
    cols = ["row", "col", "x", "y", "Aa_20yr", "log1p_Aa", "SoilType", outcome, *env]
    v = frame[cols].copy()
    continuous = ["x", "y", "log1p_Aa", outcome, *env]
    numeric = v[continuous].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    d = v.loc[keep].copy()
    tile_cells = int(config["tile_cells"])
    d["tile_row"] = np.floor(d["row"].to_numpy(dtype=float) / tile_cells).astype(np.int32)
    d["tile_col"] = np.floor(d["col"].to_numpy(dtype=float) / tile_cells).astype(np.int32)
    agg = {"x": "mean", "y": "mean", "log1p_Aa": "mean", outcome: "mean", **{x: "mean" for x in env}}
    units = d.groupby(["tile_row", "tile_col"], sort=True).agg(agg)
    units["n_pixels"] = d.groupby(["tile_row", "tile_col"], sort=True).size()
    units["SoilType"] = d.groupby(["tile_row", "tile_col"], sort=True)["SoilType"].agg(
        lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan)
    units = units.reset_index()
    categories = [int(x) for x in config["soil_type_categories"]]
    soil = pd.to_numeric(units["SoilType"], errors="coerce").to_numpy(dtype=float)
    for code in categories:
        units[f"soil_{code}"] = (soil == code).astype(float)
    units["soil_other"] = (~np.isin(soil, categories)).astype(float)
    units["soil_group"] = [str(int(s)) if np.isfinite(s) and s in categories else "other" for s in soil]
    units["outcome"] = outcome
    return units


def _weighted_mean_sd(a: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(a, axis=0, weights=w)
    var = np.average((a - mean) ** 2, axis=0, weights=w)
    return mean, np.sqrt(np.maximum(var, 1e-20))


def _prepare_features(units: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    names = ["log1p_Aa", *config["environment_covariates"],
             *[f"soil_{int(x)}" for x in config["soil_type_categories"]], "soil_other"]
    raw = units[names].to_numpy(dtype=float)
    freq = units["n_pixels"].to_numpy(dtype=float)
    mean, sd = _weighted_mean_sd(raw, freq)
    X = (raw - mean) / sd
    yraw = units[units["outcome"].iloc[0]].to_numpy(dtype=float)
    ymean = float(np.average(yraw, weights=freq))
    ysd = float(np.sqrt(np.average((yraw - ymean) ** 2, weights=freq)))
    y = (yraw - ymean) / max(ysd, 1e-12)
    meta = {"xmean": mean, "xsd": sd, "ymean": np.array([ymean]), "ysd": np.array([ysd])}
    return X, y, names, meta


def _target_indices(units: pd.DataFrame, grid_m: int) -> np.ndarray:
    xx = units["x"].to_numpy(dtype=float)
    yy = units["y"].to_numpy(dtype=float)
    gx = np.floor(xx / grid_m).astype(np.int64)
    gy = np.floor(yy / grid_m).astype(np.int64)
    d = pd.DataFrame({"i": np.arange(len(units)), "gx": gx, "gy": gy, "x": xx, "y": yy})
    centers_x = (d["gx"].to_numpy() + .5) * grid_m
    centers_y = (d["gy"].to_numpy() + .5) * grid_m
    d["center_distance"] = (d["x"] - centers_x) ** 2 + (d["y"] - centers_y) ** 2
    return (d.sort_values("center_distance").drop_duplicates(["gx", "gy"], keep="first")
            .sort_values("i")["i"].to_numpy(dtype=int))


def _ale_for_local_model(model: RandomForestRegressor, X: np.ndarray, train: np.ndarray,
                         freq: np.ndarray, exposure_index: int, bins: int) -> list[dict[str, float]]:
    e = X[train, exposure_index]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return []
    increments, counts, centers = [], [], []
    for j in range(len(edges) - 1):
        lo, hi = float(edges[j]), float(edges[j + 1])
        mask = (e >= lo) & ((e < hi) if j < len(edges) - 2 else (e <= hi))
        selected = train[mask]
        if len(selected) < 3:
            continue
        low_x = X[selected].copy(); high_x = X[selected].copy()
        low_x[:, exposure_index] = lo; high_x[:, exposure_index] = hi
        delta = model.predict(high_x) - model.predict(low_x)
        weight = freq[selected]
        increments.append(float(np.average(delta, weights=weight)))
        counts.append(float(weight.sum()))
        centers.append(float(np.average(e[mask], weights=weight)))
    if not increments:
        return []
    cumulative = np.cumsum(increments)
    center_value = float(np.average(cumulative, weights=counts))
    cumulative = cumulative - center_value
    return [{"bin_index": i, "exposure_standardized": centers[i], "ale_standardized": float(cumulative[i]),
             "bin_weight": counts[i]} for i in range(len(increments))]


def _fit_geographical_forest(units: pd.DataFrame, outcome: str, config: dict[str, Any],
                             seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, names, meta = _prepare_features(units, config)
    freq = units["n_pixels"].to_numpy(dtype=float)
    coords = units[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    targets = _target_indices(units, int(config["target_grid_m"]))
    k = min(int(config["local_rf_neighbors"]), len(units) - 1)
    dist, neighbor = tree.query(coords[targets], k=k + 1)
    metrics, importances, shap_rows, ale_rows = [], [], [], []
    key = outcome in config["shap_outcomes"]
    exp_idx = names.index("log1p_Aa")
    twi_idx, ndvi_idx = names.index("TWI"), names.index("NDVI")
    rng = np.random.default_rng(int(config["cv_seed"]) + seed_offset)
    for site_no, target in enumerate(targets):
        local_idx = neighbor[site_no, 1:]
        local_distance = dist[site_no, 1:]
        order = np.argsort(local_distance)
        local_idx = local_idx[order]
        n_valid = max(30, int(round(len(local_idx) * float(config["local_holdout_fraction"]))))
        perm = rng.permutation(len(local_idx))
        valid_idx = local_idx[perm[:n_valid]]
        train_idx = local_idx[perm[n_valid:]]
        model = RandomForestRegressor(
            n_estimators=int(config["rf_n_estimators"]),
            min_samples_leaf=int(config["rf_min_samples_leaf"]),
            max_features=float(config["rf_max_features"]),
            n_jobs=1, random_state=int(config["cv_seed"]) + seed_offset + site_no,
        )
        model.fit(X[train_idx], y[train_idx], sample_weight=freq[train_idx])
        valid_pred = model.predict(X[valid_idx])
        local_r2 = float(r2_score(y[valid_idx], valid_pred, sample_weight=freq[valid_idx]))
        local_rmse = float(np.sqrt(mean_squared_error(y[valid_idx], valid_pred, sample_weight=freq[valid_idx])))
        target_x = X[target:target + 1]
        pred_target = float(model.predict(target_x)[0])
        step = .25
        low_x, high_x = target_x.copy(), target_x.copy()
        low_x[0, exp_idx] -= step; high_x[0, exp_idx] += step
        marginal = float((model.predict(high_x)[0] - model.predict(low_x)[0]) / (2 * step))
        row = {"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0], "y": coords[target, 1],
               "soil_group": units.iloc[target]["soil_group"], "target_grid_m": int(config["target_grid_m"]),
               "local_neighbors": int(k), "local_radius_m": float(np.median(local_distance)),
               "n_local_train": int(len(train_idx)), "n_local_validation": int(len(valid_idx)),
               "local_r2_random_holdout": local_r2, "local_rmse_standardized": local_rmse,
               "target_prediction_standardized": pred_target, "target_residual_standardized": float(pred_target-y[target]),
               "local_Aa_marginal_effect_standardized": marginal,
               "outcome_guided_selection": True}
        metrics.append(row)
        for name, imp in zip(names, model.feature_importances_, strict=True):
            importances.append({"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0],
                                "y": coords[target, 1], "soil_group": units.iloc[target]["soil_group"],
                                "feature": name, "local_impurity_importance": float(imp),
                                "outcome_guided_selection": True})
        if key:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            sv_raw = explainer.shap_values(target_x, check_additivity=False)
            sv = np.asarray(getattr(sv_raw, "values", sv_raw)).squeeze()
            for name, value in zip(names, np.asarray(sv).reshape(-1), strict=True):
                shap_rows.append({"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0],
                                  "y": coords[target, 1], "soil_group": units.iloc[target]["soil_group"],
                                  "feature": name, "shap_standardized": float(value),
                                  "outcome_guided_selection": True})
            iv = np.asarray(explainer.shap_interaction_values(target_x)).squeeze()
            shap_rows.append({"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0],
                              "y": coords[target, 1], "soil_group": units.iloc[target]["soil_group"],
                              "feature": "log1p_Aa:TWI", "shap_interaction_standardized": float(iv[exp_idx, twi_idx]),
                              "outcome_guided_selection": True})
            shap_rows.append({"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0],
                              "y": coords[target, 1], "soil_group": units.iloc[target]["soil_group"],
                              "feature": "log1p_Aa:NDVI", "shap_interaction_standardized": float(iv[exp_idx, ndvi_idx]),
                              "outcome_guided_selection": True})
            for ale in _ale_for_local_model(model, X, train_idx, freq, exp_idx, int(config["ale_bins"])):
                ale_rows.append({"outcome": outcome, "target_unit_index": int(target), "x": coords[target, 0],
                                 "y": coords[target, 1], "soil_group": units.iloc[target]["soil_group"], **ale,
                                 "outcome_guided_selection": True})
        if site_no == 0 or (site_no + 1) % 50 == 0 or site_no + 1 == len(targets):
            print(f"GRF {outcome}: {site_no+1}/{len(targets)} local models", flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(importances), pd.DataFrame(shap_rows), pd.DataFrame(ale_rows)


class _SVCDesign:
    def __init__(self, units: pd.DataFrame, config: dict[str, Any]):
        self.coords = units[["x", "y"]].to_numpy(dtype=float) / 1000
        self.env_names = [*config["environment_covariates"],
                          *[f"soil_{int(x)}" for x in config["soil_type_categories"]], "soil_other"]
        env_raw = units[self.env_names].to_numpy(dtype=float)
        self.env_mean, self.env_sd = _weighted_mean_sd(env_raw, units["n_pixels"].to_numpy(dtype=float))
        self.env = (env_raw-self.env_mean)/self.env_sd
        exposure = units["log1p_Aa"].to_numpy(dtype=float)
        w = units["n_pixels"].to_numpy(dtype=float)
        self.exp_mean = float(np.average(exposure, weights=w))
        self.exp_sd = float(np.sqrt(np.average((exposure-self.exp_mean)**2, weights=w)))
        self.exposure = (exposure-self.exp_mean)/max(self.exp_sd,1e-12)
        knots = int(config["gam_spatial_knots"])
        self.sx = SplineTransformer(n_knots=knots, degree=3, knots="quantile", include_bias=False, extrapolation="linear")
        self.sy = SplineTransformer(n_knots=knots, degree=3, knots="quantile", include_bias=False, extrapolation="linear")
        self.se = SplineTransformer(n_knots=int(config["gam_exposure_knots"]), degree=3, knots="quantile",
                                    include_bias=False, extrapolation="linear")
        bx = self.sx.fit_transform(self.coords[:,0,None])
        by = self.sy.fit_transform(self.coords[:,1,None])
        self.spatial_basis = (bx[:,:,None]*by[:,None,:]).reshape(len(units),-1)
        self.exposure_basis = self.se.fit_transform(self.exposure[:,None])

    def matrix(self, exposure_std: np.ndarray | float, model_kind: str,
               indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        spatial_basis=self.spatial_basis if indices is None else self.spatial_basis[indices]
        env=self.env if indices is None else self.env[indices]
        n=len(spatial_basis)
        e = np.broadcast_to(np.asarray(exposure_std,dtype=float), (n,))
        be = self.se.transform(e[:,None])
        parts = [spatial_basis, be]
        if model_kind == "svc_gam":
            parts.append((spatial_basis[:,:,None]*be[:,None,:]).reshape(n,-1))
        parts.append(env)
        return np.column_stack(parts)


def _groups_5km(units: pd.DataFrame, block_m: int) -> np.ndarray:
    return (np.floor(units["x"].to_numpy()/block_m).astype(np.int64).astype(str)+"_"+
            np.floor(units["y"].to_numpy()/block_m).astype(np.int64).astype(str))


def _weighted_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    return {"r2": float(r2_score(y,pred,sample_weight=w)),
            "rmse": float(np.sqrt(mean_squared_error(y,pred,sample_weight=w))),
            "mae": float(mean_absolute_error(y,pred,sample_weight=w))}


def _turning_points(effects: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (outcome,unit),d in effects.groupby(["outcome","unit_index"],sort=False):
        d=d.sort_values("exposure_quantile")
        x=d["exposure_log1pAa_standardized"].to_numpy(dtype=float)
        slope=d["marginal_effect_standardized"].to_numpy(dtype=float)
        crossing=np.flatnonzero(slope[:-1]*slope[1:]<0)
        if len(crossing):
            j=int(crossing[0]);den=slope[j+1]-slope[j]
            point=x[j]-slope[j]*(x[j+1]-x[j])/den if den else np.nan
            status="sign_change"
        elif np.all(slope<0):point=np.nan;status="negative_all_quantiles"
        elif np.all(slope>0):point=np.nan;status="positive_all_quantiles"
        else:point=np.nan;status="mixed_without_bracketed_crossing"
        rows.append({"outcome":outcome,"unit_index":unit,"x":float(d["x"].iloc[0]),"y":float(d["y"].iloc[0]),
                     "turning_point_exposure_standardized":point,"response_shape_status":status,
                     "outcome_guided_selection":True})
    return pd.DataFrame(rows)


def _fit_svc_gam(units: pd.DataFrame, outcome: str, design: _SVCDesign,
                 config: dict[str, Any], seed_offset: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    yraw = units[outcome].to_numpy(dtype=float)
    w = units["n_pixels"].to_numpy(dtype=float)
    ymean = float(np.average(yraw,weights=w)); ysd=float(np.sqrt(np.average((yraw-ymean)**2,weights=w)))
    y=(yraw-ymean)/max(ysd,1e-12)
    groups=_groups_5km(units,int(config["spatial_block_size_m"]))
    outer=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=int(config["cv_seed"])+seed_offset)
    trainval,test=next(outer.split(np.zeros(len(units)),y,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=int(config["cv_seed"])+seed_offset+19)
    train_sub,val_sub=next(inner.split(np.zeros(len(trainval)),y[trainval],groups[trainval]))
    train,val=trainval[train_sub],trainval[val_sub]
    records=[]; models={}
    for model_kind in ("global_GAM", "svc_gam"):
        X=design.matrix(design.exposure,model_kind)
        scores=[]
        for alpha in config["gam_alphas"]:
            model=Ridge(alpha=float(alpha),solver="lsqr")
            model.fit(X[train],y[train],sample_weight=w[train])
            pred=model.predict(X[val])
            score=_weighted_metrics(y[val],pred,w[val])
            scores.append((float(alpha),score))
        best_alpha,best_score=max(scores,key=lambda item:item[1]["r2"])
        eval_model=Ridge(alpha=best_alpha,solver="lsqr").fit(X[trainval],y[trainval],sample_weight=w[trainval])
        test_metrics=_weighted_metrics(y[test],eval_model.predict(X[test]),w[test])
        final_model=Ridge(alpha=best_alpha,solver="lsqr").fit(X,y,sample_weight=w)
        records.append({"outcome":outcome,"model_family":model_kind,"alpha":best_alpha,
                        "n_spatial_units":len(units),"n_5km_blocks":int(pd.Series(groups).nunique()),
                        "validation_r2":best_score["r2"],"test_r2_spatial_block":test_metrics["r2"],
                        "test_rmse_standardized":test_metrics["rmse"],"test_mae_standardized":test_metrics["mae"],
                        "outcome_guided_selection":True})
        models[model_kind]=final_model
    rows=[]
    exposure_levels=np.quantile(design.exposure,[.1,.3,.5,.7,.9])
    step=.05
    for qi,level in enumerate(exposure_levels):
        plus=design.matrix(float(level+step),"svc_gam")
        minus=design.matrix(float(level-step),"svc_gam")
        deriv=(models["svc_gam"].predict(plus)-models["svc_gam"].predict(minus))/(2*step)
        for i,value in enumerate(deriv):
            rows.append({"outcome":outcome,"unit_index":i,"x":design.coords[i,0]*1000,"y":design.coords[i,1]*1000,
                         "exposure_quantile":float([.1,.3,.5,.7,.9][qi]),"exposure_log1pAa_standardized":float(level),
                         "marginal_effect_standardized":float(value),"outcome_guided_selection":True})
    return {"outcome":outcome,"models":models,"metrics":pd.DataFrame(records),"effects":pd.DataFrame(rows),
            "y_mean":ymean,"y_sd":ysd,"design":design}


def _save_map_grid(run_dir: Path, name: str, frames: list[tuple[str,pd.DataFrame,str]],
                   mask_info: tuple[np.ndarray,np.ndarray,np.ndarray], dpi: int) -> list[str]:
    mask,xcoords,ycoords=mask_info
    fig,axes=plt.subplots(1,len(frames),figsize=(7*len(frames),6),constrained_layout=True)
    axes=np.atleast_1d(axes)
    for ax,(title,d,value_col) in zip(axes,frames,strict=True):
        vals=d[value_col].to_numpy(dtype=float); finite=np.isfinite(vals)
        scale=max(float(np.quantile(np.abs(vals[finite]),.97)),.05) if finite.any() else .1
        sc=ax.scatter(d.loc[finite,"x"]/1000,d.loc[finite,"y"]/1000,c=vals[finite],s=23,marker="s",
                      linewidths=0,cmap="RdBu_r",vmin=-scale,vmax=scale,rasterized=True)
        spatial._add_map_decor(ax,mask,xcoords,ycoords)
        ax.set_title(title);ax.set_xlabel("Easting (km, EPSG:32649)");ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal",adjustable="datalim")
        fig.colorbar(sc,ax=ax,shrink=.82,label=value_col)
    base=run_dir/"figures"/name;paths=[]
    for ext in ("png","svg","pdf"):
        path=base.with_suffix(f".{ext}");fig.savefig(path,dpi=dpi,bbox_inches="tight");paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _save_ale_figure(run_dir: Path, ale: pd.DataFrame, dpi: int) -> list[str]:
    if ale.empty:return []
    summaries=[]
    for (outcome,soil,bin_idx),d in ale.groupby(["outcome","soil_group","bin_index"],sort=True):
        w=d["bin_weight"].to_numpy(dtype=float)
        summaries.append({"outcome":outcome,"soil_group":soil,"bin_index":bin_idx,
                          "exposure":float(np.average(d["exposure_standardized"],weights=w)),
                          "ale":float(np.average(d["ale_standardized"],weights=w)),
                          "n_sites":int(d["target_unit_index"].nunique())})
    summary=pd.DataFrame(summaries)
    colors=SOIL_COLORS
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=False,constrained_layout=True)
    for ax,outcome in zip(axes,KEY_OUTCOMES,strict=True):
        d=summary[summary.outcome.eq(outcome)]
        for soil,g in d.groupby("soil_group"):
            ax.plot(g.exposure,g.ale,marker="o",lw=1.8,label=f"土类 {soil}",color=colors.get(str(soil),colors["other"]))
        ax.axhline(0,color="#6b7280",lw=.8);ax.set_title(LABELS[outcome]);ax.set_xlabel("log1p(Aa) 标准化值")
        ax.set_ylabel("局部 ALE（标准化健康响应）");ax.grid(color="#d8dee3",lw=.5);ax.legend(fontsize=8)
    paths=[];base=run_dir/"figures"/"local_ale_by_soil_type"
    for ext in ("png","svg","pdf"):
        path=base.with_suffix(f".{ext}");fig.savefig(path,dpi=dpi,bbox_inches="tight");paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    summary.to_csv(run_dir/"tables"/"local_ale_by_soil_type.csv",index=False)
    return paths


def _save_shap_interaction_figure(run_dir: Path, shap_rows: pd.DataFrame, dpi: int) -> list[str]:
    if shap_rows.empty:return []
    rows=shap_rows[shap_rows["feature"].isin(["log1p_Aa:TWI","log1p_Aa:NDVI"])].copy()
    fig,axes=plt.subplots(3,2,figsize=(12,13),constrained_layout=True)
    for i,outcome in enumerate(KEY_OUTCOMES):
        d=rows[rows.outcome.eq(outcome)]
        for j,(feature,color_col) in enumerate([("log1p_Aa:TWI","TWI"),("log1p_Aa:NDVI","NDVI")]):
            ax=axes[i,j];g=d[d.feature.eq(feature)]
            sc=ax.scatter(g.x/1000,g.y/1000,c=g["shap_interaction_standardized"],s=17,cmap="RdBu_r",alpha=.85)
            ax.set_title(f"{LABELS[outcome]} × {color_col} interaction SHAP")
            ax.set_xlabel("Easting (km)");ax.set_ylabel("Northing (km)");fig.colorbar(sc,ax=ax,shrink=.78)
    paths=[];base=run_dir/"figures"/"spatial_shap_interactions"
    for ext in ("png","svg","pdf"):
        path=base.with_suffix(f".{ext}");fig.savefig(path,dpi=dpi,bbox_inches="tight");paths.append(str(path.relative_to(ROOT)))
    plt.close(fig);return paths


def _save_svc_curves(run_dir: Path, curve: pd.DataFrame, dpi: int) -> list[str]:
    if curve.empty:return []
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=False,constrained_layout=True)
    for ax,outcome in zip(axes,KEY_OUTCOMES,strict=True):
        d=curve[curve.outcome.eq(outcome)]
        for site,g in d.groupby("site_role"):
            ax.plot(g.exposure_log1pAa_standardized,g.predicted_health_standardized,marker=None,lw=2,label=site)
        ax.set_title(f"{LABELS[outcome]} 局部 SVC-GAM 响应曲线")
        ax.set_xlabel("log1p(Aa) 标准化值");ax.set_ylabel("预测健康值（标准化）");ax.grid(color="#d8dee3",lw=.5);ax.legend(fontsize=8)
    paths=[];base=run_dir/"figures"/"svc_gam_local_curves"
    for ext in ("png","svg","pdf"):
        path=base.with_suffix(f".{ext}");fig.savefig(path,dpi=dpi,bbox_inches="tight");paths.append(str(path.relative_to(ROOT)))
    plt.close(fig);return paths


def _report(run_dir: Path, metrics: pd.DataFrame, gam_metrics: pd.DataFrame, effects: pd.DataFrame,
            turning: pd.DataFrame, importance: pd.DataFrame, shap_long: pd.DataFrame,
            shap_interactions: pd.DataFrame, ale: pd.DataFrame,
            target_count: int, env_rows: int) -> str:
    lines=["# Phase I-5：地理加权随机森林、空间 SHAP/ALE 与 SVC-GAM","",
           f"权威运行：`{run_dir.relative_to(ROOT)}`。运行模式为 `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。","",
           "## 样本与建模尺度","",
           f"- 全库耕地掩膜为 392,527 行；环境完整且 Aa 有效像元为 {env_rows:,} 行。对九项健康结局按逐项完整案例汇总为约 240 m 单元，局部模型使用像元频数权重。",
           f"- 地理加权 RF 在每个约 {int(1200)} m 网格选一个代表位置，共 {target_count:,} 个局部中心；每个局部森林使用最近 300 个汇总单元，按 75/25 随机拆分拟合与局部验证。相邻验证单元仍有空间依赖，R²不是空间独立测试。",
           "- RF 自变量包含 log1p(Aa_20yr)、九项非 RUSLE 环境连续协变量及 SoilType one-hot（26、27、42、85、其他）；不读 R/K/LS/C/P。重点 SHI、OM、SW 额外计算局部 TreeSHAP、Aa×TWI/Aa×NDVI SHAP 交互和 ALE。",
           "- SVC-GAM 使用空间坐标与侵蚀暴露的张量积 B-spline，分别比较全局暴露平滑项和空间变化暴露平滑项；按 5 km 空间块做验证与一次测试，最终用全体环境完整空间单元重拟合制图。",
           "", "## 地理加权 RF","",
           "| 指标 | 局部验证 R² 中位数 | R² P10–P90 | 局部 Aa 边际效应中位数 | 局部 SHAP Aa 中位数 |",
           "|---|---:|---:|---:|---:|"]
    shap_aa=shap_long[(shap_long.feature.eq("log1p_Aa")) & shap_long.get("shap_standardized",pd.Series(index=shap_long.index,dtype=float)).notna()] if not shap_long.empty else pd.DataFrame()
    for outcome in KEY_OUTCOMES:
        d=metrics[metrics.outcome.eq(outcome)]
        s=shap_aa[shap_aa.outcome.eq(outcome)] if not shap_aa.empty else pd.DataFrame()
        lines.append(f"| {LABELS[outcome]} | {d.local_r2_random_holdout.median():.3f} | {d.local_r2_random_holdout.quantile(.1):.3f}–{d.local_r2_random_holdout.quantile(.9):.3f} | {d.local_Aa_marginal_effect_standardized.median():.3f} | {s.shap_standardized.median():.3f} |")
    lines.extend(["", "局部 RF 的局部 R²表示局部随机留出预测，不代表侵蚀胁迫强度。局部变量重要性表使用 impurity importance；侵蚀效应另由边际预测差、SHAP 和 ALE 描述。SHAP 值是随机森林对预测值的分解，不是因果贡献。",
                  "", "| 指标 | 局部 impurity importance 均值前三项 |", "|---|---|"])
    for outcome in KEY_OUTCOMES:
        imp=(importance[importance.outcome.eq(outcome)].groupby("feature").local_impurity_importance.mean()
             .sort_values(ascending=False).head(3))
        lines.append(f"| {LABELS[outcome]} | "+"、".join(f"{k} ({v:.3f})" for k,v in imp.items())+" |")
    soil85 = ale.loc[pd.to_numeric(ale.soil_group, errors="coerce").eq(85)] if not ale.empty else pd.DataFrame()
    soil85_by_outcome = soil85.groupby("outcome").target_unit_index.nunique() if not soil85.empty else pd.Series(dtype=int)
    soil85_range = f"{int(soil85_by_outcome.min())}–{int(soil85_by_outcome.max())}" if len(soil85_by_outcome) else "0"
    lines.extend(["", "表中 Aa SHAP 的正负是相对局部模型基线的预测贡献，不能与有限差分边际效应的方向直接互换。",
                  "", "## SHAP、交互与 ALE","",
                  f"- 重点指标共输出 {len(shap_aa):,} 个局部 Aa SHAP 值；每个空间中心计算 Aa×TWI 和 Aa×NDVI TreeSHAP 交互，并输出空间散点图。交互 SHAP 描述模型预测分解中的交互，不作因果解释。",
                  f"- 局部 ALE 曲线覆盖 {ale.target_unit_index.nunique() if not ale.empty else 0:,} 个空间中心，按中心所在代表土类 26、27、42、85、其他分层汇总；曲线用局部森林邻域样本构造，不能替代匹配或实测对照。",
                  f"- SoilType=85 的 ALE 每项重点指标仅有 {soil85_range} 个代表位置，样本支持不足，不据此比较土类响应。",
                  "", "## 空间变化系数 GAM","",
                  "| 指标 | 模型 | 验证 R² | 5 km 整块测试 R² | 测试 RMSE | alpha |",
                  "|---|---|---:|---:|---:|---:|"])
    for r in gam_metrics.itertuples(index=False):
        lines.append(f"| {LABELS[r.outcome]} | {r.model_family} | {r.validation_r2:.3f} | {r.test_r2_spatial_block:.3f} | {r.test_rmse_standardized:.3f} | {r.alpha:g} |")
    base=gam_metrics[gam_metrics.model_family.eq("global_GAM")].set_index("outcome")
    svc=gam_metrics[gam_metrics.model_family.eq("svc_gam")].set_index("outcome")
    lines.append("")
    for outcome in KEY_OUTCOMES:
        delta=float(svc.loc[outcome,"test_r2_spatial_block"]-base.loc[outcome,"test_r2_spatial_block"])
        lines.append(f"- **{LABELS[outcome]}**：加入空间变化暴露曲面后，整块测试 R² 改变量为 {delta:+.3f}；相对全局暴露平滑项的增益较小，局部曲面主要作为解释性探索。")
    median=effects[effects.exposure_quantile.eq(.5)] if not effects.empty else pd.DataFrame()
    for outcome in KEY_OUTCOMES:
        d=median[median.outcome.eq(outcome)]
        if len(d):
            lines.append(f"- **{LABELS[outcome]} 中位暴露边际效应**：空间单元均值 {d.marginal_effect_standardized.mean():.3f}，负向单元占比 {(d.marginal_effect_standardized<0).mean():.1%}。")
        t=turning[turning.outcome.eq(outcome)] if not turning.empty else pd.DataFrame()
        if len(t):
            cross=t.response_shape_status.eq("sign_change").mean()
            lines.append(f"- **{LABELS[outcome]} 局部转折诊断**：在 q10–q90 评估范围内 {cross:.1%} 的空间单元出现有限差分斜率变号；这只是模型曲线诊断，未作为阈值推断。")
    lines.extend(["", "## 解释限制","",
                  "- 环境完整分析总体、健康预测栅格、240 m 汇总和空间相关性限制均沿用 I-4。RF 局部随机留出受邻近像元依赖影响；空间泛化以 SVC-GAM 的 5 km 整块测试单独报告。",
                  "- SVC-GAM 是基于 sklearn B-spline 张量积与 Ridge 惩罚的空间变化系数近似，不是 mgcv/pyGAM 的 GAM 实现；局部效应由标准化暴露上的有限差分计算。",
                  "- 局部森林中心间隔约 1.2 km，地图显示的是局部模型中心，不是 30 m 独立模型。SoilType=85 保留；其水体含义与耕地掩膜关系存在项目既有歧义。",
                  "- 健康数据为随机森林预测面，像元和局部窗口不等于独立实测样本量。本轮未重算或验证 RUSLE 因子乘积，未运行 PCNM，未修改研究总纲。",
                  "", "图件与表格见 `figures/`、`tables/` 和 `manifests/`。"," "])
    path=run_dir/"reports"/"PHASE_I_SPATIAL_ML_REPORT.md";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines),encoding="utf-8");return str(path.relative_to(ROOT))


def run(config_path: Path) -> Path:
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir=ROOT/config["output_root"]/config["run_id"]
    if run_dir.exists():raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for sub in ("configs","tables","figures","reports","manifests","leaderboards"):(run_dir/sub).mkdir(exist_ok=True)
    (run_dir/"configs"/config_path.name).write_text(config_path.read_text(encoding="utf-8"),encoding="utf-8")
    pixels,env_path,health_path=_read_pixels(config)
    full_env=np.isfinite(pixels[["log1p_Aa",*config["environment_covariates"]]].to_numpy(dtype=float)).all(axis=1)
    mask_info=spatial._mask_arrays(pixels)
    grf_metrics_all=[];importance_all=[];shap_all=[];ale_all=[];units_by_outcome={}
    for oi,outcome in enumerate(config["outcomes"]):
        valid=np.isfinite(pd.to_numeric(pixels[outcome],errors="coerce")).to_numpy() & full_env
        units=_aggregate_pixels(pixels.loc[valid],outcome,config)
        units_by_outcome[outcome]=units
        print(f"Spatial ML {outcome}: {len(units):,} block units",flush=True)
        metrics,importance,shap_rows,ale=_fit_geographical_forest(units,outcome,config,oi*10000)
        grf_metrics_all.append(metrics);importance_all.append(importance)
        if not shap_rows.empty:shap_all.append(shap_rows)
        if not ale.empty:ale_all.append(ale)
        metrics.to_csv(run_dir/"tables"/f"grf_local_metrics_{outcome}.csv",index=False)
    grf_metrics=pd.concat(grf_metrics_all,ignore_index=True)
    importances=pd.concat(importance_all,ignore_index=True)
    shap_rows=pd.concat(shap_all,ignore_index=True) if shap_all else pd.DataFrame()
    ale=pd.concat(ale_all,ignore_index=True) if ale_all else pd.DataFrame()
    grf_metrics.to_csv(run_dir/"tables"/"grf_local_metrics.csv",index=False)
    importances.to_parquet(run_dir/"tables"/"grf_local_feature_importance.parquet",index=False)
    shap_rows.to_parquet(run_dir/"tables"/"spatial_shap_values_and_interactions.parquet",index=False)
    ale.to_csv(run_dir/"tables"/"local_ale_curves_by_site.csv",index=False)
    # SVC-GAM comparison and effect surfaces.
    gam_metrics=[];gam_effects=[];gam_curves=[];gam_map_frames=[]
    for oi,outcome in enumerate(config["outcomes"]):
        units=units_by_outcome[outcome]
        design=_SVCDesign(units,config)
        result=_fit_svc_gam(units,outcome,design,config,oi*31)
        gam_metrics.append(result["metrics"]);effects=result["effects"];gam_effects.append(effects)
        metrics=result["metrics"]
        svc_metrics=metrics[metrics.model_family.eq("svc_gam")].iloc[0]
        # Recreate final full-data model to produce representative local response curves.
        model=result["models"]["svc_gam"]
        egrid=np.linspace(np.quantile(design.exposure,.05),np.quantile(design.exposure,.95),60)
        med=effects[effects.exposure_quantile.eq(.5)].sort_values("marginal_effect_standardized")
        if outcome in config["key_outcomes"] and len(med):
            chosen=[("强负向",med.iloc[0]),("中位",med.iloc[len(med)//2]),("弱负向或正向",med.iloc[-1])]
            for role,row in chosen:
                idx=int(row.unit_index)
                for e in egrid:
                    xm=design.matrix(float(e),"svc_gam",indices=[idx])
                    pred=float(model.predict(xm)[0])
                    gam_curves.append({"outcome":outcome,"unit_index":idx,"x":design.coords[idx,0]*1000,
                                       "y":design.coords[idx,1]*1000,"site_role":role,
                                       "exposure_log1pAa_standardized":float(e),"predicted_health_standardized":pred})
        if outcome in config["key_outcomes"]:
            q50=effects[effects.exposure_quantile.eq(.5)].copy()
            q50["local_svc_gam_marginal_effect_q50"] = q50["marginal_effect_standardized"]
            gam_map_frames.append((outcome,q50))
        print(f"SVC-GAM {outcome}: spatial-block test R2={svc_metrics.test_r2_spatial_block:.3f}",flush=True)
    gam_metrics_df=pd.concat(gam_metrics,ignore_index=True)
    gam_effects_df=pd.concat(gam_effects,ignore_index=True)
    turning_df=_turning_points(gam_effects_df)
    gam_curves_df=pd.DataFrame(gam_curves)
    gam_metrics_df.to_csv(run_dir/"tables"/"svc_gam_model_comparison.csv",index=False)
    gam_effects_df.to_parquet(run_dir/"tables"/"svc_gam_local_marginal_effects.parquet",index=False)
    turning_df.to_csv(run_dir/"tables"/"svc_gam_local_turning_points.csv",index=False)
    gam_curves_df.to_csv(run_dir/"tables"/"svc_gam_representative_curves.csv",index=False)
    grf_leader=[]
    for outcome,d in grf_metrics.groupby("outcome"):
        sl=d["local_Aa_marginal_effect_standardized"]
        grf_leader.append({"outcome":outcome,"exposure":"log1p(Aa_20yr)","analysis_population":"environment_complete_block_means",
                           "model_family":"GeographicalRandomForest","environment_adjustment":"nine continuous covariates + SoilType one-hot",
                           "matching_scheme":"none","environment_group":"none","bandwidth":int(config["local_rf_neighbors"]),
                           "spatial_scale_m":int(config["target_grid_m"]),"n_samples":int(len(d)),
                           "r2":float(d["local_r2_random_holdout"].median()),"adjusted_r2":np.nan,
                           "rmse":float(d["local_rmse_standardized"].median()),"mae":np.nan,"spearman":np.nan,
                           "global_slope":float(sl.mean()),"local_coefficient_summary":f"median={sl.median():.4f}; p10={sl.quantile(.1):.4f}; p90={sl.quantile(.9):.4f}",
                           "negative_effect_ratio":float((sl<0).mean()),"local_r2":float(d["local_r2_random_holdout"].median()),
                           "validation_r2":float(d["local_r2_random_holdout"].median()),"test_r2_spatial":np.nan,
                           "experiment_id":config["run_id"],"outcome_guided_selection":True})
    gam_leader=[]
    effect_q50=gam_effects_df[gam_effects_df.exposure_quantile.eq(.5)]
    for r in gam_metrics_df.itertuples(index=False):
        e=effect_q50[effect_q50.outcome.eq(r.outcome)]
        gam_leader.append({"outcome":r.outcome,"exposure":"log1p(Aa_20yr)","analysis_population":"environment_complete_block_means",
                           "model_family":r.model_family,"environment_adjustment":"nine continuous covariates + SoilType one-hot",
                           "matching_scheme":"none","environment_group":"none","bandwidth":np.nan,"spatial_scale_m":5000,
                           "n_samples":r.n_spatial_units,"r2":r.test_r2_spatial_block,"adjusted_r2":np.nan,
                           "rmse":r.test_rmse_standardized,"mae":r.test_mae_standardized,"spearman":np.nan,
                           "global_slope":float(e.marginal_effect_standardized.mean()) if len(e) and r.model_family=="svc_gam" else np.nan,
                           "local_coefficient_summary":f"alpha={r.alpha:g}",
                           "negative_effect_ratio":float((e.marginal_effect_standardized<0).mean()) if len(e) and r.model_family=="svc_gam" else np.nan,
                           "local_r2":np.nan,"validation_r2":r.validation_r2,"test_r2_spatial":r.test_r2_spatial_block,
                           "experiment_id":config["run_id"],"outcome_guided_selection":True})
    spatial_ml_leaderboard=pd.DataFrame(grf_leader+gam_leader)
    spatial_ml_leaderboard.to_csv(run_dir/"leaderboards"/"phase_i_spatial_ml_leaderboard.csv",index=False)
    # Map template and compact model-specific figures.
    mask_info=spatial._mask_arrays(pixels)
    paths=[]
    for name,column in [("grf_local_r2","local_r2_random_holdout"),("grf_local_Aa_marginal_effect","local_Aa_marginal_effect_standardized")]:
        frames=[(LABELS[o],grf_metrics[grf_metrics.outcome.eq(o)],column) for o in KEY_OUTCOMES]
        paths+=_save_map_grid(run_dir,name,frames,mask_info,int(config["map_dpi"]))
    shap_aa=shap_rows[(shap_rows.feature.eq("log1p_Aa")) & shap_rows.get("shap_standardized",pd.Series(index=shap_rows.index,dtype=float)).notna()] if not shap_rows.empty else pd.DataFrame()
    frames=[(LABELS[o],shap_aa[shap_aa.outcome.eq(o)],"shap_standardized") for o in KEY_OUTCOMES]
    paths+=_save_map_grid(run_dir,"spatial_shap_Aa",frames,mask_info,int(config["map_dpi"]))
    for outcome,d in gam_map_frames:
        d["x"]=d["x"].to_numpy(dtype=float);d["y"]=d["y"].to_numpy(dtype=float)
    paths+=_save_map_grid(run_dir,"svc_gam_marginal_effect_q50",
                          [(LABELS[o],d,"local_svc_gam_marginal_effect_q50") for o,d in gam_map_frames],
                          mask_info,int(config["map_dpi"]))
    paths+=_save_ale_figure(run_dir,ale,int(config["map_dpi"]))
    paths+=_save_shap_interaction_figure(run_dir,shap_rows,int(config["map_dpi"]))
    paths+=_save_svc_curves(run_dir,gam_curves_df,int(config["map_dpi"]))
    package={name:_version(name) for name in ("shap","scikit-learn","scipy","pandas","numpy","statsmodels")}
    report=_report(run_dir,grf_metrics,gam_metrics_df,gam_effects_df,turning_df,importances,shap_rows,shap_rows,ale,
                   len(_target_indices(units_by_outcome["SHI_candidate"],int(config["target_grid_m"]))),int(full_env.sum()))
    provenance={"phase":"I-5","mode":"EXPLORATORY_ORACLE","outcome_guided_selection":True,
                "run_id":config["run_id"],"created_utc":datetime.now(timezone.utc).isoformat(),
                "git_commit":_git("rev-parse","HEAD"),"git_status_porcelain":_git("status","--short"),
                "inputs":{"environment_erosion":str(env_path.relative_to(ROOT)),"health":str(health_path.relative_to(ROOT))},
                "configuration":str((run_dir/"configs"/config_path.name).relative_to(ROOT)),
                "rows":{"mask":len(pixels),"environment_and_exposure_complete":int(full_env.sum()),
                        "local_centers":len(_target_indices(units_by_outcome["SHI_candidate"],int(config["target_grid_m"])))},
                "analysis_unit":"240 m block means; local RF centers on 1.2 km grid; SVC-GAM 240 m units",
                "coordinate_reference":"EPSG:32649","software":{"python":platform.python_version(),**package},
                "methods":{"grf":"local RF on nearest 300 block means; local random holdout is not spatial-block validation",
                           "shap":"TreeSHAP local contribution and Aa×TWI/Aa×NDVI interaction on key outcomes",
                           "svc_gam":"spatial tensor-product B-splines with Ridge penalty; 5 km block holdout"},
                "outputs":[str(x.relative_to(ROOT)) for x in [run_dir/"tables"/"grf_local_metrics.csv",
                  run_dir/"tables"/"grf_local_feature_importance.parquet",run_dir/"tables"/"spatial_shap_values_and_interactions.parquet",
                  run_dir/"tables"/"local_ale_curves_by_site.csv",run_dir/"tables"/"svc_gam_model_comparison.csv",
                  run_dir/"tables"/"svc_gam_local_marginal_effects.parquet",run_dir/"tables"/"svc_gam_local_turning_points.csv",
                  run_dir/"reports"/"PHASE_I_SPATIAL_ML_REPORT.md"]]+paths,
                "report":report}
    (run_dir/"manifests"/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    (run_dir/"RUN_STATUS.md").write_text("# Run status\n\nStatus: authoritative exploratory result\n\n- Phase: I-5\n- `outcome_guided_selection=true`\n- Local RF holdout is random within local neighborhoods; only SVC-GAM reports 5 km spatial-block test metrics.\n- SHAP and ALE describe model predictions, not causal effects.\n",encoding="utf-8")
    files=sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file() and p.name!="output_manifest.json")
    (run_dir/"manifests"/"output_manifest.json").write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding="utf-8")
    return run_dir


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=ROOT/"config"/"phase_i_spatial_ml.yaml")
    args=parser.parse_args();run_dir=run(args.config)
    print(run_dir.relative_to(ROOT));print((run_dir/"reports"/"PHASE_I_SPATIAL_ML_REPORT.md").relative_to(ROOT))


if __name__=="__main__":main()
