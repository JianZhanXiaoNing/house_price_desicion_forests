# ==============================================
# 独立脚本：优化稀有类别误判（仅过采样，稳定版）
# 输入：上一轮优化模型（optimized_rf_model.pkl）、原始数据（train.csv）
# 输出：最终优化模型（final_optimized_model.pkl）+ 优化报告
# 核心修改：移除易出错的特征加权，仅保留稀有样本过采样（效果明确、无索引问题）
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
# 步骤1：加载资源（模型、数据、参数）
# ==============================================
def load_resources(model_path="C:/Users/19734/Desktop/model_improved/optimized_rf_model.pkl",
                   data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv"):
    """加载模型、数据，提取预处理逻辑和参数"""
    print("=" * 60)
    print("步骤1：加载原模型和数据...")
    print("=" * 60)

    # 1. 加载上一轮优化后的模型
    model = joblib.load(model_path)
    print(f"模型加载成功（类型：{type(model)}）")

    # 2. 加载原始数据，分离X和y
    df = pd.read_csv(data_path)
    X = df.drop(["Id", "SalePrice"], axis=1)
    y = df["SalePrice"]
    print(f"数据加载成功：特征数{X.shape[1]}，样本数{X.shape[0]}")

    # 3. 提取关键信息（复用预处理逻辑和模型参数）
    preprocessor = model.named_steps["preprocessor"]
    model_params = model.named_steps["regressor"].get_params()
    print(f"复用模型参数：{model_params}")

    # 4. 划分训练集/测试集（与原流程一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. 评估原模型性能（基准）
    def evaluate(model, X, y):
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return {"RMSE": round(rmse, 2), "R²": round(r2, 4)}

    base_perf = evaluate(model, X_test, y_test)
    print(f"\n基准模型性能（优化前）：RMSE={base_perf['RMSE']}美元，R²={base_perf['R²']}")

    # 6. 识别稀有类别样本（核心：用于过采样和效果验证）
    rare_samples_idx = identify_rare_samples(X_train)
    print(f"训练集中稀有类别样本数：{len(rare_samples_idx)}（占比：{len(rare_samples_idx) / len(X_train):.2%}）")

    # 7. 识别测试集中的稀有样本（用于验证优化效果）
    rare_samples_test_idx = identify_rare_samples(X_test)
    print(f"测试集中稀有类别样本数：{len(rare_samples_test_idx)}（占比：{len(rare_samples_test_idx) / len(X_test):.2%}）")

    return (
        model, preprocessor, model_params,
        X, y, X_train, X_test, y_train, y_test,
        base_perf, rare_samples_idx, rare_samples_test_idx
    )


# ==============================================
# 步骤2：核心工具函数 - 识别稀有类别和样本
# ==============================================
def identify_rare_samples(X, rare_threshold=0.05):
    """识别稀有类别样本：分类特征中占比<rare_threshold的类别对应的样本"""
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    rare_samples_idx = set()

    for feat in categorical_features:
        # 计算每个类别的占比（排除NaN，避免将缺失值误判为稀有类别）
        feat_value_counts = X[feat].value_counts(normalize=True, dropna=True)
        # 识别稀有类别（占比<阈值）
        rare_categories = feat_value_counts[feat_value_counts < rare_threshold].index.tolist()
        if rare_categories:
            # 记录稀有类别对应的样本索引（包含NaN？不，仅稀有类别）
            rare_idx = X[X[feat].isin(rare_categories)].index.tolist()
            rare_samples_idx.update(rare_idx)
            print(f"特征{feat}的稀有类别：{rare_categories}（每个类别占比<5%）")

    return list(rare_samples_idx)


def oversample_rare_samples(X_train, y_train, rare_samples_idx, oversample_ratio=2):
    """过采样稀有类别样本：将稀有样本复制oversample_ratio倍，提升其在训练集中的占比"""
    print(f"\n开始过采样：原始稀有样本数{len(rare_samples_idx)}，放大{oversample_ratio}倍")

    # 提取稀有样本和普通样本（保留原始索引，避免数据混乱）
    X_rare = X_train.loc[rare_samples_idx].copy()
    y_rare = y_train.loc[rare_samples_idx].copy()
    X_normal = X_train.drop(rare_samples_idx).copy()
    y_normal = y_train.drop(rare_samples_idx).copy()

    # 过采样稀有样本（复制+重置索引，避免索引重复）
    X_rare_oversampled = pd.concat([X_rare] * oversample_ratio, ignore_index=True)
    y_rare_oversampled = pd.concat([y_rare] * oversample_ratio, ignore_index=True)

    # 合并过采样后的训练集（普通样本 + 过采样稀有样本）
    X_train_oversampled = pd.concat([X_normal, X_rare_oversampled], ignore_index=True)
    y_train_oversampled = pd.concat([y_normal, y_rare_oversampled], ignore_index=True)

    # 打乱训练集（避免稀有样本集中在末尾，影响模型训练）
    shuffle_idx = np.random.permutation(len(X_train_oversampled))
    X_train_oversampled = X_train_oversampled.iloc[shuffle_idx].reset_index(drop=True)
    y_train_oversampled = y_train_oversampled.iloc[shuffle_idx].reset_index(drop=True)

    print(
        f"过采样完成：总样本数{len(X_train_oversampled)}，稀有样本占比{len(X_rare_oversampled) / len(X_train_oversampled):.2%}")
    return X_train_oversampled, y_train_oversampled


# ==============================================
# 步骤3：训练优化模型（仅过采样，稳定可靠）
# ==============================================
def train_rare_optimized_model(
        preprocessor, model_params,
        X_train, y_train, X_test, y_test,
        rare_samples_idx, rare_samples_test_idx
):
    """训练优化模型：仅稀有样本过采样（移除易出错的特征加权）"""
    print("\n" + "=" * 60)
    print("步骤3：训练稀有类别优化模型（仅过采样）...")
    print("=" * 60)

    # 1. 过采样稀有类别样本（核心优化步骤）
    X_train_oversampled, y_train_oversampled = oversample_rare_samples(
        X_train, y_train, rare_samples_idx, oversample_ratio=2  # 放大2倍，可调整
    )

    # 2. 构建优化后的Pipeline（仅复用原预处理+回归器，无额外加权步骤）
    optimized_model = Pipeline(steps=[
        ("preprocessor", preprocessor),  # 复用原预处理逻辑（避免索引问题）
        ("regressor", RandomForestRegressor(**model_params))  # 复用原参数
    ])

    # 3. 训练模型（用过量采样后的训练集）
    optimized_model.fit(X_train_oversampled, y_train_oversampled)

    # 4. 评估优化后模型性能（整体+稀有样本）
    # 整体性能
    optimized_perf = evaluate(optimized_model, X_test, y_test)
    print(f"\n优化后模型整体性能：RMSE={optimized_perf['RMSE']}美元，R²={optimized_perf['R²']}")

    # 稀有样本性能（核心验证指标）
    optimized_rare_perf = None
    if rare_samples_test_idx:
        X_test_rare = X_test.loc[rare_samples_test_idx]
        y_test_rare = y_test.loc[rare_samples_test_idx]
        optimized_rare_perf = evaluate(optimized_model, X_test_rare, y_test_rare)
        print(f"优化后模型稀有样本性能：RMSE={optimized_rare_perf['RMSE']}美元，R²={optimized_rare_perf['R²']}")

    return optimized_model, optimized_perf, optimized_rare_perf


# ==============================================
# 步骤4：保存结果与性能对比
# ==============================================
def save_results(
        optimized_model, base_perf, optimized_perf, optimized_rare_perf,
        rare_samples_idx, rare_samples_test_idx
):
    """保存最终模型和优化报告"""
    print("\n" + "=" * 60)
    print("步骤4：保存结果...")
    print("=" * 60)

    # 1. 保存最终优化模型
    joblib.dump(optimized_model, "final_optimized_model.pkl")
    print("最终优化模型已保存为：final_optimized_model.pkl")

    # 2. 生成性能对比报告
    perf_data = {
        "指标": ["整体RMSE（美元）", "整体R²（拟合优度）"],
        "优化前（基准）": [base_perf["RMSE"], base_perf["R²"]],
        "优化后（稀有样本过采样）": [optimized_perf["RMSE"], optimized_perf["R²"]],
        "变化量": [
            round(optimized_perf["RMSE"] - base_perf["RMSE"], 2),
            round(optimized_perf["R²"] - base_perf["R²"], 4)
        ],
        "变化率": [
            f"{round((optimized_perf['RMSE'] - base_perf['RMSE']) / base_perf['RMSE'] * 100, 2)}%",
            f"{round((optimized_perf['R²'] - base_perf['R²']) / base_perf['R²'] * 100, 2)}%"
        ]
    }

    # 补充稀有样本性能对比（若有）
    if optimized_rare_perf:
        # 计算基准模型在稀有样本上的性能（重新训练基准模型，确保公平）
        base_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**model_params))
        ])
        base_model.fit(X_train, y_train)
        base_rare_perf = evaluate(base_model, X_test.loc[rare_samples_test_idx], y_test.loc[rare_samples_test_idx])

        perf_data["指标"].extend(["稀有样本RMSE（美元）", "稀有样本R²"])
        perf_data["优化前（基准）"].extend([base_rare_perf["RMSE"], base_rare_perf["R²"]])
        perf_data["优化后（稀有样本过采样）"].extend([optimized_rare_perf["RMSE"], optimized_rare_perf["R²"]])
        perf_data["变化量"].extend([
            round(optimized_rare_perf["RMSE"] - base_rare_perf["RMSE"], 2),
            round(optimized_rare_perf["R²"] - base_rare_perf["R²"], 4)
        ])
        perf_data["变化率"].extend([
            f"{round((optimized_rare_perf['RMSE'] - base_rare_perf['RMSE']) / base_rare_perf['RMSE'] * 100, 2)}%",
            f"{round((optimized_rare_perf['R²'] - base_rare_perf['R²']) / base_rare_perf['R²'] * 100, 2)}%"
        ])

    perf_compare = pd.DataFrame(perf_data)
    perf_compare.to_csv("稀有类别优化性能对比.csv", index=False, encoding="utf-8-sig")
    print("性能对比报告已保存为：稀有类别优化性能对比.csv")

    # 3. 生成优化报告
    optimization_report = {
        "优化策略": ["稀有类别过采样（放大2倍）"],
        "训练集稀有样本数（优化前）": [len(rare_samples_idx)],
        "训练集稀有样本数（优化后）": [len(rare_samples_idx) * 2],
        "测试集稀有样本数": [len(rare_samples_test_idx)],
        "整体RMSE变化": [f"{perf_compare.iloc[0]['变化量']}美元（{perf_compare.iloc[0]['变化率']}）"],
        "整体R²变化": [f"{perf_compare.iloc[1]['变化量']}（{perf_compare.iloc[1]['变化率']}）"]
    }
    if optimized_rare_perf:
        optimization_report["稀有样本RMSE变化"] = [
            f"{perf_compare.iloc[2]['变化量']}美元（{perf_compare.iloc[2]['变化率']}）"]

    report_df = pd.DataFrame(optimization_report)
    report_df.to_csv("稀有类别优化报告.csv", index=False, encoding="utf-8-sig")
    print("优化报告已保存为：稀有类别优化报告.csv")

    # 打印最终总结
    print("\n" + "=" * 60)
    print("最终优化总结：")
    print("=" * 60)
    print(perf_compare.to_string(index=False))
    if optimized_perf["RMSE"] < base_perf["RMSE"]:
        print(f"\n✅ 稀有类别优化成功！整体RMSE降低{abs(perf_compare.iloc[0]['变化量'])}美元")
        if optimized_rare_perf and optimized_rare_perf["RMSE"] < base_rare_perf["RMSE"]:
            print(
                f"✅ 稀有样本预测效果显著提升！RMSE降低{abs(perf_compare.iloc[2]['变化量'])}美元（{abs(float(perf_compare.iloc[2]['变化率'].replace('%', '')))}%）")
    else:
        print(f"\n❌ 优化未达预期，可调整过采样倍数（当前：2倍），建议尝试3倍或1.5倍")


# ==============================================
# 辅助函数：评估模型
# ==============================================
def evaluate(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return {"RMSE": round(rmse, 2), "R²": round(r2, 4)}


# ==============================================
# 主函数：一键执行优化流程
# ==============================================
if __name__ == "__main__":
    # 1. 加载资源（上一轮强相关优化后的模型）
    (
        base_model, preprocessor, model_params,
        X, y, X_train, X_test, y_train, y_test,
        base_perf, rare_samples_idx, rare_samples_test_idx
    ) = load_resources(
        model_path="C:/Users/19734/Desktop/model_improved/optimized_rf_model.pkl",  # 上一轮优化后的模型路径
        data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv"  # 数据路径
    )

    # 2. 训练稀有类别优化模型（仅过采样）
    optimized_model, optimized_perf, optimized_rare_perf = train_rare_optimized_model(
        preprocessor=preprocessor,
        model_params=model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        rare_samples_idx=rare_samples_idx,
        rare_samples_test_idx=rare_samples_test_idx
    )

    # 3. 保存结果
    save_results(
        optimized_model=optimized_model,
        base_perf=base_perf,
        optimized_perf=optimized_perf,
        optimized_rare_perf=optimized_rare_perf,
        rare_samples_idx=rare_samples_idx,
        rare_samples_test_idx=rare_samples_test_idx
    )