# Ames房屋售价预测：决策森林模型完整代码
# 适用场景：基于Ames数据集构建房价回归模型，含数据预处理、模型训练、调优与评估
# 依赖库：pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
# 数据文件：需将train.csv放在代码同级目录，或修改文件路径

# ==============================================
# 一、环境准备与库导入
# ==============================================
# 基础数据处理库
import pandas as pd
import numpy as np

# 数据预处理库
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 模型训练与评估库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 可视化库（特征重要性分析）
import matplotlib.pyplot as plt
import seaborn as sns

# 模型保存库
import joblib

# 忽略警告信息
import warnings

warnings.filterwarnings("ignore")


# ==============================================
# 二、数据加载与初步探索
# ==============================================
def load_and_explore_data(data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv"):
    """
    加载数据并执行初步探索：查看数据规模、缺失值、分离特征与目标变量
    参数：data_path - 数据文件路径
    返回：X_train, X_test, y_train, y_test - 划分后的训练集/测试集
    """
    # 1. 加载数据
    print("=" * 50)
    print("步骤1：加载数据并初步探索")
    print("=" * 50)
    df = pd.read_csv(data_path)
    print(f"数据形状（行数×列数）：{df.shape}")
    print("\n前5行数据预览：")
    print(df.head())

    # 2. 查看缺失值比例（仅显示缺失率>0的特征）
    missing_ratio = df.isnull().sum() / len(df)
    missing_features = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
    print("\n缺失值比例（缺失率>0的特征）：")
    print(missing_features)

    # 3. 分离特征（X）和目标变量（y）：排除Id（无预测意义）
    X = df.drop(["Id", "SalePrice"], axis=1)
    y = df["SalePrice"]
    print(f"\n特征矩阵形状：{X.shape}，目标变量形状：{y.shape}")

    # 4. 划分训练集与测试集（8:2分割，固定随机种子确保可复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n训练集规模：{X_train.shape}，测试集规模：{X_test.shape}")
    print(f"训练集目标变量均值：{y_train.mean():.2f} 美元")
    print(f"测试集目标变量均值：{y_test.mean():.2f} 美元")

    return X_train, X_test, y_train, y_test, X


# 执行数据加载
X_train, X_test, y_train, y_test, X = load_and_explore_data(data_path="C:/Users/19734/Desktop/Ames/house-prices-advanced-regression-techniques/train.csv")  # 若路径不同，需修改此处


# ==============================================
# 三、数据预处理Pipeline设计
# ==============================================
def create_preprocessor(X):
    """
    构建数据预处理流水线：区分数值/分类特征，分别处理缺失值与编码
    参数：X - 原始特征矩阵（用于识别特征类型）
    返回：preprocessor - 预处理流水线
    """
    print("\n" + "=" * 50)
    print("步骤2：构建数据预处理流水线")
    print("=" * 50)

    # 1. 区分特征类型：分类特征（object）与数值特征（int/float）
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"分类特征数量：{len(categorical_features)}，示例：{categorical_features[:5]}")
    print(f"数值特征数量：{len(numerical_features)}，示例：{numerical_features[:5]}")

    # 2. 数值特征预处理：中位数填充缺失值（抗异常值）
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))  # 数值特征缺失值用中位数填充
        # ("scaler", StandardScaler())  # 决策森林对量纲不敏感，可省略标准化
    ])

    # 3. 分类特征预处理："Missing"填充缺失值 + 独热编码
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),  # 缺失值标记为"Missing"
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # 忽略测试集未知类别
    ])

    # 4. 整合预处理流程：对不同类型特征应用不同处理
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    print("\n预处理流水线构建完成：")
    print(f"- 数值特征：中位数填充")
    print(f"- 分类特征：'Missing'填充 + 独热编码")
    return preprocessor


# 构建预处理流水线
preprocessor = create_preprocessor(X)


# ==============================================
# 四、基础决策森林模型训练与评估
# ==============================================
def train_baseline_model(preprocessor, X_train, y_train, X_test, y_test):
    """
    训练基础决策森林模型，执行交叉验证与测试集评估
    参数：preprocessor - 预处理流水线；X_train/X_test - 训练/测试特征；y_train/y_test - 训练/测试目标
    返回：baseline_pipeline - 基础模型流水线
    """
    print("\n" + "=" * 50)
    print("步骤3：训练基础决策森林模型")
    print("=" * 50)

    # 1. 构建完整流水线：预处理 + 决策森林
    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_estimators=100))
    ])

    # 2. 5折交叉验证（评估模型稳定性）
    print("\n执行5折交叉验证...")
    cv_r2 = cross_val_score(baseline_pipeline, X_train, y_train, cv=5, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(baseline_pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"))

    print(f"\n5折交叉验证结果：")
    print(f"平均R²：{cv_r2.mean():.4f}（±{cv_r2.std():.4f}）")
    print(f"平均RMSE：{cv_rmse.mean():.2f} 美元（±{cv_rmse.std():.2f}）")

    # 3. 训练模型并在测试集评估
    baseline_pipeline.fit(X_train, y_train)
    y_pred = baseline_pipeline.predict(X_test)

    # 计算测试集指标
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"\n测试集评估结果（基础模型）：")
    print(f"R²：{test_r2:.4f}（越接近1越好，表示模型解释力）")
    print(f"RMSE：{test_rmse:.2f} 美元（平均预测误差）")
    print(f"MAE：{test_mae:.2f} 美元（平均绝对误差）")

    return baseline_pipeline


# 训练基础模型
baseline_model = train_baseline_model(preprocessor, X_train, y_train, X_test, y_test)


# ==============================================
# 五、模型优化（网格搜索调参）
# ==============================================
def optimize_model(preprocessor, X_train, y_train, X_test, y_test):
    """
    用网格搜索优化决策森林参数，获取最优模型
    参数：preprocessor - 预处理流水线；X_train/X_test - 训练/测试特征；y_train/y_test - 训练/测试目标
    返回：best_model - 最优模型
    """
    print("\n" + "=" * 50)
    print("步骤4：网格搜索优化模型参数")
    print("=" * 50)

    # 1. 定义参数网格（根据计算资源调整范围，避免过大）
    param_grid = {
        "regressor__n_estimators": [100, 200, 300],  # 决策树数量
        "regressor__max_depth": [None, 10, 20, 30],  # 树最大深度（None表示不限制）
        "regressor__min_samples_split": [2, 5, 10],  # 分裂所需最小样本数
        "regressor__min_samples_leaf": [1, 2, 4]  # 叶子节点最小样本数
    }
    print(f"\n待搜索参数网格：")
    for param, values in param_grid.items():
        print(f"- {param}: {values}")

    # 2. 构建网格搜索对象（5折交叉验证，基于R²优化）
    grid_search = GridSearchCV(
        estimator=Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))]),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,  # 利用所有CPU核心加速
        verbose=1  # 显示搜索进度
    )

    # 3. 执行网格搜索
    print("\n开始网格搜索（耗时较长，耐心等待...）")
    grid_search.fit(X_train, y_train)

    # 4. 输出最优参数与交叉验证结果
    print(f"\n最优参数组合：")
    for param, value in grid_search.best_params_.items():
        print(f"- {param}: {value}")
    print(f"\n最优参数下5折交叉验证R²：{grid_search.best_score_:.4f}")

    # 5. 最优模型测试集评估
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    best_test_r2 = r2_score(y_test, y_pred_best)
    best_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
    best_test_mae = mean_absolute_error(y_test, y_pred_best)

    print(f"\n最优模型测试集评估结果：")
    print(
        f"R²：{best_test_r2:.4f}（较基础模型提升：{best_test_r2 - r2_score(y_test, baseline_model.predict(X_test)):.4f}）")
    print(
        f"RMSE：{best_test_rmse:.2f} 美元（较基础模型降低：{np.sqrt(mean_squared_error(y_test, baseline_model.predict(X_test))) - best_test_rmse:.2f}）")
    print(f"MAE：{best_test_mae:.2f} 美元")

    return best_model, grid_search.best_params_


# 执行模型优化（若计算资源有限，可跳过此步，直接使用基础模型）
best_model, best_params = optimize_model(preprocessor, X_train, y_train, X_test, y_test)


# ==============================================
# 六、特征重要性分析（可视化）
# ==============================================
def analyze_feature_importance(best_model, X):
    """
    分析并可视化决策森林的特征重要性，输出Top20重要特征（兼容所有版本，无报错）
    参数：best_model - 最优模型；X - 原始特征矩阵（用于获取特征名）
    """
    print("\n" + "=" * 50)
    print("步骤5：特征重要性分析与可视化")
    print("=" * 50)

    # 1. 区分原始特征类型
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 2. 获取预处理后的特征名（独热编码后的分类特征 + 原始数值特征）
    cat_encoder = best_model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = list(numerical_features) + list(cat_feature_names)

    # 3. 获取特征重要性并排序
    feature_importance = best_model.named_steps["regressor"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False).head(20)  # 取Top20

    # 4. 可视化Top20重要特征（用matplotlib原生实现，兼容性100%）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(12, 8))

    # 核心：matplotlib原生barh绘制水平条形图
    ax.barh(
        y=range(len(importance_df)),  # y轴用索引（避免特征名过长错位）
        width=importance_df["importance"],
        color="#2E86AB",
        alpha=0.8  # 透明度，更美观
    )

    # 设置y轴标签为特征名（对应索引位置）
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"], fontsize=10)

    # 图表美化（保持原风格）
    ax.set_xlabel("特征重要性", fontsize=12, fontweight="bold")
    ax.set_ylabel("特征名称", fontsize=12, fontweight="bold")
    ax.set_title("Ames房价预测：Top20重要特征", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # 重要性最高的特征在顶部
    ax.grid(axis="x", alpha=0.3, linestyle="--")  # 横向网格线，更易读

    # 调整布局，避免特征名被截断
    plt.tight_layout()
    # 保存图片（高清晰度）
    plt.savefig("feature_importance_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. 输出Top10重要特征
    print("\nTop10重要特征（影响房价的核心因素）：")
    print(importance_df.head(10).to_string(index=False))
    print(f"\n特征重要性可视化图已保存为：feature_importance_top20.png")


# 执行特征重要性分析
analyze_feature_importance(best_model, X)


# ==============================================
# 七、模型保存（后续复用）
# ==============================================
def save_model(best_model, save_path="ames_house_price_rf_model.pkl"):
    """
    保存最优模型到本地，便于后续预测新数据
    参数：best_model - 最优模型；save_path - 保存路径
    """
    print("\n" + "=" * 50)
    print("步骤6：保存最优模型")
    print("=" * 50)

    # 保存模型
    joblib.dump(best_model, save_path)
    print(f"最优模型已保存至：{save_path}")

    # 示例：加载模型的代码（后续使用时）
    print(f"\n后续加载模型并预测新数据的代码：")
    print(f"import joblib")
    print(f"loaded_model = joblib.load('{save_path}')")
    print(f"new_data = pd.read_csv('new_house_data.csv')  # 新房屋数据")
    print(f"new_data = new_data.drop('Id', axis=1) if 'Id' in new_data.columns else new_data")
    print(f"new_price_pred = loaded_model.predict(new_data)")
    print(f"print('新房屋预测售价：', new_price_pred)")


# 保存模型
save_model(best_model)





# ==============================================
# 错误模式检测：收集错误样本+分类标注
# 依赖库：pandas, numpy, scikit-learn, matplotlib, seaborn
# 输入：训练好的最优模型（best_model）、原始数据集（X_train/X_test/y_train/y_test/X）
# 输出：1. 错误样本分类文件 2. 各错误模式误差可视化 3. 错误模式统计报告
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def detect_error_patterns(best_model, X_train, X_test, y_train, y_test, X, save_path="C:/Users/19734/Desktop/mistakes_sorted/error_analysis/"):
    """
    检测模型错误模式，收集错误样本并分类标注
    参数：
        best_model: 训练好的最优决策森林模型
        X_train/X_test: 训练/测试特征
        y_train/y_test: 训练/测试目标变量
        X: 原始特征矩阵（用于特征类型识别）
        save_path: 结果保存路径
    返回：
        error_samples_df: 带错误模式标注的完整样本数据框
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    print("="*60)
    print("开始模型错误模式检测...")
    print("="*60)

    # 步骤1：生成全量样本的预测结果（训练集+测试集，便于扩大错误样本池）
    print("\n步骤1：生成全量样本预测结果...")
    # 合并训练集和测试集
    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    # 生成预测值和误差
    y_pred_full = best_model.predict(X_full)
    # 计算绝对误差和相对误差（相对误差更能反映错误程度）
    error_abs = np.abs(y_full - y_pred_full)  # 绝对误差
    error_rel = error_abs / y_full  # 相对误差（相对于真实售价）

    # 构建全量样本数据框（含预测结果和误差）
    full_df = X_full.copy()
    full_df["真实售价"] = y_full
    full_df["预测售价"] = y_pred_full.round(2)
    full_df["绝对误差"] = error_abs.round(2)
    full_df["相对误差"] = error_rel.round(4)
    full_df["数据类型"] = ["训练集"]*len(X_train) + ["测试集"]*len(X_test)

    # 定义错误样本阈值：相对误差>0.2（20%）或绝对误差>30000美元（根据数据集调整）
    error_threshold_rel = 0.2
    error_threshold_abs = 30000
    full_df["是否错误样本"] = (full_df["相对误差"] > error_threshold_rel) | (full_df["绝对误差"] > error_threshold_abs)
    error_samples_df = full_df[full_df["是否错误样本"]].copy()

    print(f"全量样本总数：{len(full_df)}")
    print(f"错误样本数：{len(error_samples_df)}")
    print(f"错误样本占比：{len(error_samples_df)/len(full_df):.2%}")

    # 步骤2：定义错误模式的判断规则（基于特征和误差特点）
    print("\n步骤2：按错误模式分类标注错误样本...")

    # 2.1 错误模式1：极端房价偏差（极高/极低房价）
    # 按真实售价分5段，取首尾两段为极端值
    price_quantiles = y_full.quantile([0.2, 0.4, 0.6, 0.8]).values
    low_price_thres = price_quantiles[0]  # 低价位阈值（20分位数）
    high_price_thres = price_quantiles[-1]  # 高价位阈值（80分位数）
    error_samples_df["错误模式_极端房价"] = (error_samples_df["真实售价"] < low_price_thres) | (error_samples_df["真实售价"] > high_price_thres)

    # 2.2 错误模式2：稀有类别误判（基于稀有特征）
    # 定义稀有特征及稀有类别（占比<5%为稀有）
    rare_features = {
        "PoolQC": ["Ex", "Gd", "TA"],  # 有泳池（无泳池为多数）
        "SaleCondition": ["Abnorml", "AdjLand", "Alloca"],  # 异常交易
        "BldgType": ["2fmCon", "Duplex", "TwnhsE"],  # 非独栋住宅
        "GarageType": ["CarPort", "2Types"],  # 特殊车库类型
        "CentralAir": ["N"]  # 无中央空调（多数有）
    }
    error_samples_df["错误模式_稀有类别"] = False
    for feat, rare_cats in rare_features.items():
        if feat in error_samples_df.columns:
            error_samples_df["错误模式_稀有类别"] = error_samples_df["错误模式_稀有类别"] | error_samples_df[feat].isin(rare_cats)

    # 2.3 错误模式3：强相关特征冗余（面积类/车库类相关特征）
    # 强相关特征组（相关系数>0.7）
    corr_feature_groups = [
        ("GrLivArea", "1stFlrSF"),  # 地上居住面积与一层面积
        ("GarageArea", "GarageCars"),  # 车库面积与车位数
        ("TotalBsmtSF", "BsmtFinSF1"),  # 地下室总面积与成品面积
        ("YearBuilt", "YearRemodAdd")  # 建造年份与翻新年份
    ]
    error_samples_df["错误模式_强相关特征"] = False
    for feat1, feat2 in corr_feature_groups:
        if feat1 in error_samples_df.columns and feat2 in error_samples_df.columns:
            # 计算样本级别的特征相关性（标准化后乘积）
            def normalize_feat(series):
                return (series - series.mean()) / series.std()
            feat1_norm = normalize_feat(error_samples_df[feat1])
            feat2_norm = normalize_feat(error_samples_df[feat2])
            # 乘积绝对值>1表示两特征同时偏离均值，存在冗余干扰
            error_samples_df["错误模式_强相关特征"] = error_samples_df["错误模式_强相关特征"] | (np.abs(feat1_norm * feat2_norm) > 1)

    # 2.4 错误模式4：缺失值集中
    # 定义高缺失率特征（原始缺失率>10%）
    high_missing_features = ["LotFrontage", "GarageYrBlt", "MasVnrArea", "BsmtFinSF2", "BsmtUnfSF"]
    error_samples_df["缺失特征数"] = error_samples_df[high_missing_features].apply(lambda x: sum(x.isin(["Missing", np.nan])), axis=1)
    error_samples_df["错误模式_缺失值集中"] = error_samples_df["缺失特征数"] >= 2  # 2个及以上高缺失特征

    # 2.5 错误模式5：数据分布偏移（时间/社区）
    # 时间分布偏移：销售年份的极端值（首尾10%）
    year_quantiles = full_df["YrSold"].quantile([0.1, 0.9]).values
    early_year_thres = year_quantiles[0]
    late_year_thres = year_quantiles[1]
    # 社区分布偏移：训练集中占比<3%的社区
    neighborhood_counts = X_train["Neighborhood"].value_counts(normalize=True)
    rare_neighborhoods = neighborhood_counts[neighborhood_counts < 0.03].index.tolist()
    error_samples_df["错误模式_分布偏移"] = (error_samples_df["YrSold"] < early_year_thres) | (error_samples_df["YrSold"] > late_year_thres) | error_samples_df["Neighborhood"].isin(rare_neighborhoods)

    # 步骤3：统计各错误模式的分布
    print("\n步骤3：错误模式统计结果：")
    error_pattern_cols = [col for col in error_samples_df.columns if col.startswith("错误模式_")]
    pattern_stats = {}
    for pattern in error_pattern_cols:
        count = error_samples_df[pattern].sum()
        ratio = count / len(error_samples_df)
        pattern_stats[pattern.replace("错误模式_", "")] = {"数量": count, "占错误样本比": f"{ratio:.2%}"}
        print(f"- {pattern.replace('错误模式_', '')}：{count}个（{ratio:.2%}）")

    # 处理多模式重叠样本
    error_samples_df["错误模式_多重叠加"] = error_samples_df[error_pattern_cols].sum(axis=1) >= 2
    multi_pattern_count = error_samples_df["错误模式_多重叠加"].sum()
    print(f"- 多重错误模式叠加：{multi_pattern_count}个（{multi_pattern_count/len(error_samples_df):.2%}）")

    # 步骤4：保存错误样本明细和统计报告
    print("\n步骤4：保存结果文件...")
    # 保存错误样本明细（含所有特征和错误模式标注）
    error_samples_df.to_csv(f"{save_path}错误样本分类明细.csv", index=False, encoding="utf-8-sig")
    # 保存错误模式统计报告
    pattern_stats_df = pd.DataFrame(pattern_stats).T
    pattern_stats_df["占全样本比"] = pattern_stats_df["数量"] / len(full_df)
    pattern_stats_df.to_csv(f"{save_path}错误模式统计报告.csv", encoding="utf-8-sig")

    # 步骤5：可视化错误模式分布
    print("\n步骤5：生成错误模式可视化图表...")
    # 5.1 各错误模式数量柱状图
    plt.figure(figsize=(12, 6))
    pattern_names = [name.replace("错误模式_", "") for name in error_pattern_cols]
    pattern_counts = [error_samples_df[col].sum() for col in error_pattern_cols]
    plt.bar(pattern_names, pattern_counts, color=["#E74C3C", "#3498DB", "#F39C12", "#2ECC71", "#9B59B6"])
    plt.xlabel("错误模式", fontsize=12, fontweight="bold")
    plt.ylabel("错误样本数量", fontsize=12, fontweight="bold")
    plt.title("各错误模式的错误样本分布", fontsize=14, fontweight="bold")
    plt.xticks(rotation=15)
    # 在柱子上添加数值标签
    for i, count in enumerate(pattern_counts):
        plt.text(i, count+1, str(count), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}错误模式数量分布.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.2 不同房价区间的误差分布
    full_df["房价区间"] = pd.cut(
        full_df["真实售价"],
        bins=[0, low_price_thres, price_quantiles[1], price_quantiles[2], high_price_thres, np.inf],
        labels=["极低价位", "低价位", "中价位", "高价位", "极高价位"]
    )
    error_by_price = full_df.groupby("房价区间")["相对误差"].agg(["mean", "count"]).reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="房价区间", y="mean", data=error_by_price, palette="viridis")
    plt.xlabel("房价区间", fontsize=12, fontweight="bold")
    plt.ylabel("平均相对误差", fontsize=12, fontweight="bold")
    plt.title("不同房价区间的平均相对误差", fontsize=14, fontweight="bold")
    # 添加样本数量标签
    for i, row in error_by_price.iterrows():
        plt.text(i, row["mean"]+0.01, f"样本数：{int(row['count'])}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_path}不同房价区间误差分布.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.3 错误模式重叠情况可视化（热力图）
    pattern_corr = error_samples_df[error_pattern_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pattern_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
    plt.title("错误模式之间的相关性（重叠程度）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}错误模式重叠相关性.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "="*60)
    print("错误模式检测完成！生成文件如下：")
    print(f"1. {save_path}错误样本分类明细.csv - 所有错误样本的特征+错误模式标注")
    print(f"2. {save_path}错误模式统计报告.csv - 各模式的数量和占比统计")
    print(f"3. {save_path}错误模式数量分布.png - 各模式数量柱状图")
    print(f"4. {save_path}不同房价区间误差分布.png - 房价区间vs误差")
    print(f"5. {save_path}错误模式重叠相关性.png - 模式重叠热力图")
    print("="*60)

    return error_samples_df, pattern_stats_df

# ==============================================
# 执行错误模式检测（需在训练好best_model后调用）
# ==============================================
# 注意：需确保前面的代码已生成 best_model、X_train、X_test、y_train、y_test、X
if __name__ == "__main__":
    # 调用错误模式检测函数
    error_samples, pattern_stats = detect_error_patterns(
        best_model=best_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X=X,
        save_path="C:/Users/19734/Desktop/mistakes_sorted/error_analysis/"  # 结果保存路径，可自定义
    )