# ==============================================
# 独立脚本：优化原模型（pkl文件）- 移除强相关特征冗余
# 功能：加载原模型→筛选强相关特征→重新训练→保存优化后模型
# 输入：原模型文件（ames_house_price_rf_model.pkl）、原始数据（train.csv）
# 输出：优化后模型（optimized_rf_model.pkl）+ 筛选报告
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================
# 步骤1：加载原模型、原始数据
# ==============================================
def load_original_resources(model_path="C:/Users/19734/Desktop/Tensorflow_forests/ames_house_price_rf_model.pkl", data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv"):
    """加载原模型和原始数据，提取关键信息（预处理逻辑、参数）"""
    print("=" * 60)
    print("步骤1：加载原模型和原始数据...")
    print("=" * 60)

    # 1. 加载原训练好的模型
    original_model = joblib.load(model_path)
    print(f"原模型加载成功（类型：{type(original_model)}）")

    # 2. 加载原始数据，分离特征X和目标y（与原建模一致）
    df = pd.read_csv(data_path)
    X_original = df.drop(["Id", "SalePrice"], axis=1)
    y_original = df["SalePrice"]
    print(f"原始数据加载成功：特征数{X_original.shape[1]}，样本数{X_original.shape[0]}")

    # 3. 提取原模型的关键信息（复用预处理逻辑和参数，确保公平对比）
    # 提取原预处理流程（数值/分类特征的处理逻辑）
    original_preprocessor = original_model.named_steps["preprocessor"]
    # 提取原模型的参数（如n_estimators、max_depth等）
    original_params = original_model.named_steps["regressor"].get_params()
    print(f"原模型参数：{original_params}")

    # 4. 划分训练集/测试集（与原建模一致：8:2，random_state=42）
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42
    )

    # 5. 评估原模型在测试集的性能（作为对比基准）
    def evaluate(model, X, y):
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return {"RMSE": round(rmse, 2), "R²": round(r2, 4)}

    original_perf = evaluate(original_model, X_test, y_test)
    print(f"\n原模型测试集性能：RMSE={original_perf['RMSE']}美元，R²={original_perf['R²']}")

    return (
        original_model, original_preprocessor, original_params,
        X_original, y_original, X_train, X_test, y_train, y_test, original_perf
    )


# ==============================================
# 步骤2：核心函数 - 移除强相关特征
# ==============================================
def remove_correlated_features(X, y, corr_threshold=0.7, importance_threshold=0.5):
    """移除强相关冗余特征：仅删除强相关组中的冗余特征，保留所有非强相关特征"""
    print("\n" + "=" * 60)
    print("步骤2：筛选强相关冗余特征...")
    print("=" * 60)

    # 1. 分离数值特征和分类特征（仅处理数值特征的相关性）
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    X_numerical = X[numerical_features].copy()
    print(f"数值特征总数：{len(numerical_features)}，分类特征总数：{len(categorical_features)}")

    # 2. 识别高相关特征组（相关系数>阈值）
    corr_matrix = X_numerical.corr(method="pearson")
    high_corr_groups = []
    visited = set()

    for i, feat1 in enumerate(numerical_features):
        if feat1 in visited:
            continue
        current_group = [feat1]
        for j, feat2 in enumerate(numerical_features[i + 1:]):
            if abs(corr_matrix.loc[feat1, feat2]) > corr_threshold:
                current_group.append(feat2)
                visited.add(feat2)
        if len(current_group) >= 2:
            high_corr_groups.append(current_group)
            visited.add(feat1)

    # 输出高相关组信息
    if not high_corr_groups:
        print("未发现强相关特征组，无需优化")
        return X, X.columns.tolist(), {}

    print(f"识别到{len(high_corr_groups)}组强相关特征：")
    corr_report = {}
    deleted_features = []  # 仅记录强相关组中被删除的冗余特征
    for idx, group in enumerate(high_corr_groups, 1):
        # 获取组内特征的相关矩阵，计算最大相关系数
        group_corr_matrix = corr_matrix.loc[group, group]
        triu_indices = np.triu_indices_from(group_corr_matrix, k=1)
        max_corr = group_corr_matrix.values[triu_indices].max()
        corr_report[f"组{idx}"] = {"特征列表": group, "最大相关系数": round(max_corr, 3)}
        print(f"- 组{idx}：{group}（最大相关系数：{round(max_corr, 3)}）")

    # 3. 基于特征重要性筛选强相关组内的冗余特征（仅在组内删除，不影响其他特征）
    temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_numerical_filled = X_numerical.fillna(X_numerical.median())
    temp_model.fit(X_numerical_filled, y)

    feat_importance = pd.DataFrame({
        "feature": numerical_features,
        "importance": temp_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # 保留的数值特征：强相关组中筛选后的特征 + 非强相关组的所有数值特征
    selected_numerical = []
    # 先收集所有强相关组中的特征（用于后续排除被删除的冗余特征）
    all_correlated_features = [feat for group in high_corr_groups for feat in group]

    # 处理强相关组：每组保留核心特征
    for idx, group in enumerate(high_corr_groups, 1):
        group_importance = feat_importance[feat_importance["feature"].isin(group)]
        group_total_importance = group_importance["importance"].sum()
        group_importance["ratio"] = group_importance["importance"] / group_total_importance
        kept = group_importance[group_importance["ratio"] > importance_threshold]["feature"].tolist()
        if not kept:
            kept = [group_importance.iloc[0]["feature"]]  # 至少保留1个核心特征
        deleted = [f for f in group if f not in kept]
        deleted_features.extend(deleted)
        selected_numerical.extend(kept)
        corr_report[f"组{idx}"].update({"保留特征": kept, "删除特征": deleted})
        print(f"- 组{idx}：保留{kept}，删除{deleted}")

    # 补充非强相关的数值特征（这部分之前被误删，现在重新加入）
    non_correlated_numerical = [feat for feat in numerical_features if feat not in all_correlated_features]
    selected_numerical.extend(non_correlated_numerical)
    print(f"\n非强相关数值特征（全部保留）：{non_correlated_numerical}")

    # 4. 整合最终特征：筛选后的数值特征 + 全部分类特征（分类特征不做删除）
    selected_features = selected_numerical + categorical_features
    X_filtered = X[selected_features].copy()

    # 输出筛选结果（确保只删除强相关冗余特征）
    print(f"\n筛选结果：")
    print(f"- 原始特征总数：{len(X.columns)}")
    print(f"- 筛选后特征总数：{len(selected_features)}")
    print(f"- 仅删除强相关冗余特征：{deleted_features}（共{len(deleted_features)}个）")

    # 可视化高相关特征重要性（仅显示强相关组）
    plt.figure(figsize=(12, 6))
    for idx, group in enumerate(high_corr_groups[:3]):  # 最多显示3组
        group_importance = feat_importance[feat_importance["feature"].isin(group)]
        plt.subplot(1, 3, idx + 1)
        sns.barplot(x="feature", y="importance", data=group_importance)
        plt.title(f"组{idx + 1}特征重要性（红色=删除，绿色=保留）")
        plt.xticks(rotation=45)
        # 给删除的特征标红，保留的标绿
        for i, feat in enumerate(group_importance["feature"]):
            color = "red" if feat in deleted_features else "green"
            plt.bar(i, group_importance[group_importance["feature"] == feat]["importance"].values[0], color=color)
    plt.tight_layout()
    plt.savefig("correlated_feature_importance.png", dpi=300)
    print("特征重要性对比图已保存（红色=删除，绿色=保留）")

    return X_filtered, selected_features, corr_report


# ==============================================
# 步骤3：用筛选后的特征重新训练优化模型
# ==============================================
def train_optimized_model(X_filtered, y, original_preprocessor, original_params, X_train, X_test, y_train, y_test):
    """复用原模型的预处理逻辑和参数，仅替换输入特征"""
    print("\n" + "=" * 60)
    print("步骤3：训练优化后模型...")
    print("=" * 60)

    # 1. 重构预处理流程（适配筛选后的特征）
    categorical_features = X_filtered.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 复用原预处理逻辑（数值/分类特征的填充、编码方式）
    numerical_transformer = original_preprocessor.transformers_[0][1]
    categorical_transformer = original_preprocessor.transformers_[1][1]

    optimized_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    # 2. 训练模型（完全复用原模型参数，确保对比公平）
    optimized_model = Pipeline(steps=[
        ("preprocessor", optimized_preprocessor),
        ("regressor", RandomForestRegressor(**original_params))  # 复用原参数
    ])
    optimized_model.fit(X_train[X_filtered.columns], y_train)  # 仅用筛选后的特征

    # 3. 评估优化后模型性能
    def evaluate(model, X, y):
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return {"RMSE": round(rmse, 2), "R²": round(r2, 4)}

    optimized_perf = evaluate(optimized_model, X_test[X_filtered.columns], y_test)
    print(f"优化后模型测试集性能：RMSE={optimized_perf['RMSE']}美元，R²={optimized_perf['R²']}")

    return optimized_model, optimized_perf


# ==============================================
# 步骤4：保存结果与性能对比
# ==============================================
def save_results(optimized_model, corr_report, original_perf, optimized_perf):
    """保存优化后模型、筛选报告、性能对比"""
    print("\n" + "=" * 60)
    print("步骤4：保存结果...")
    print("=" * 60)

    # 1. 保存优化后模型（覆盖或新增文件名，避免覆盖原模型）
    joblib.dump(optimized_model, "optimized_rf_model.pkl")
    print("优化后模型已保存为：optimized_rf_model.pkl")

    # 2. 保存强相关特征筛选报告
    corr_report_df = pd.DataFrame.from_dict(corr_report, orient="index")
    corr_report_df.to_csv("强相关特征筛选报告.csv", encoding="utf-8-sig")
    print("筛选报告已保存为：强相关特征筛选报告.csv")

    # 3. 保存性能对比报告
    perf_compare = pd.DataFrame({
        "指标": ["RMSE（美元）", "R²（拟合优度）"],
        "原模型": [original_perf["RMSE"], original_perf["R²"]],
        "优化后模型": [optimized_perf["RMSE"], optimized_perf["R²"]],
        "变化量": [
            round(optimized_perf["RMSE"] - original_perf["RMSE"], 2),
            round(optimized_perf["R²"] - original_perf["R²"], 4)
        ],
        "变化率": [
            f"{round((optimized_perf['RMSE'] - original_perf['RMSE']) / original_perf['RMSE'] * 100, 2)}%",
            f"{round((optimized_perf['R²'] - original_perf['R²']) / original_perf['R²'] * 100, 2)}%"
        ]
    })
    perf_compare.to_csv("模型优化性能对比.csv", index=False, encoding="utf-8-sig")
    print("性能对比报告已保存为：模型优化性能对比.csv")

    # 打印最终对比结果
    print("\n" + "=" * 60)
    print("最终性能对比总结：")
    print("=" * 60)
    print(perf_compare.to_string(index=False))
    if optimized_perf["RMSE"] < original_perf["RMSE"]:
        print(f"\n✅ 优化成功！RMSE降低{abs(perf_compare.iloc[0]['变化量'])}美元，R²提升{perf_compare.iloc[1]['变化量']}")
    else:
        print(f"\n❌ 优化未达预期，可调整corr_threshold阈值重试（当前0.7）")


# ==============================================
# 主函数：一键执行优化流程
# ==============================================
if __name__ == "__main__":
    # 1. 加载原模型和数据
    (
        original_model, original_preprocessor, original_params,
        X_original, y_original, X_train, X_test, y_train, y_test, original_perf
    ) = load_original_resources(
        model_path="C:/Users/19734/Desktop/Tensorflow_forests/ames_house_price_rf_model.pkl",  # 你的原模型路径
        data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv"  # 你的原始数据路径
    )

    # 2. 筛选强相关特征
    X_filtered, selected_features, corr_report = remove_correlated_features(
        X=X_original,
        y=y_original,
        corr_threshold=0.7,  # 可调整：0.6-0.8
        importance_threshold=0.5  # 可调整：0.3-0.6
    )

    # 3. 训练优化模型
    optimized_model, optimized_perf = train_optimized_model(
        X_filtered=X_filtered,
        y=y_original,
        original_preprocessor=original_preprocessor,
        original_params=original_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # 4. 保存结果
    save_results(optimized_model, corr_report, original_perf, optimized_perf)