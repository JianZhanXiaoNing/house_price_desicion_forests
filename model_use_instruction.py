# ==============================================
# 模型使用示例（预测新数据）
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")
def predict_new_data(model_path="ames_house_price_rf_model.pkl", new_data_path=None):
    """
    示例：加载模型并预测新房屋数据的售价
    参数：model_path - 模型保存路径；new_data_path - 新数据路径（可选）
    """
    print("\n" + "=" * 50)
    print("步骤7：模型预测新数据示例")
    print("=" * 50)

    if new_data_path is None:
        print("\n提示：若需预测新数据，请提供新数据文件路径（格式与train.csv一致），示例代码如下：")
        print(f"predict_new_data(model_path='ames_house_price_rf_model.pkl', new_data_path='new_house_data.csv')")
        return

    # 加载模型和新数据
    loaded_model = joblib.load(model_path)
    new_data = pd.read_csv(new_data_path)

    # 数据预处理（与训练时一致：排除Id，若存在）
    if "Id" in new_data.columns:
        new_data_ids = new_data["Id"]
        new_data = new_data.drop("Id", axis=1)
    else:
        new_data_ids = range(1, len(new_data) + 1)

    # 预测售价
    new_pred = loaded_model.predict(new_data)

    # 输出结果
    result_df = pd.DataFrame({
        "Id": new_data_ids,
        "Predicted_SalePrice": new_pred.round(2)
    })
    print(f"\n新数据预测结果（共{len(result_df)}条记录）：")
    print(result_df.head(10).to_string(index=False))

    # 保存预测结果
    result_df.to_csv("new_house_price_pred.csv", index=False)
    print(f"\n预测结果已保存至：new_house_price_pred.csv")


# 预测新数据示例（若有新数据，取消注释并修改路径）
# predict_new_data(model_path="ames_house_price_rf_model.pkl", new_data_path="new_house_data.csv")

print("\n" + "=" * 50)
print("Ames房屋售价预测决策森林模型构建完成！")
print("核心输出文件：")
print("1. ames_house_price_rf_model.pkl - 最优模型文件")
print("2. feature_importance_top20.png - 特征重要性可视化图")
print("3. new_house_price_pred.csv - 新数据预测结果（若执行预测）")
print("=" * 50)