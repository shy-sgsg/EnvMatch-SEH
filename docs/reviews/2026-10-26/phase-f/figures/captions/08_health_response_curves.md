# 08_health_response_curves

灰蓝色散点来自固定的10,000行绘图样本；每项结局使用其完整观测。橙色虚线为该样本的自然样条描述曲线。其他指标的蓝色实线沿用历史全样本 Oracle 择优表观模型。CEC 面板的散点与样条使用校正公式 `CEC_corrected=(CEC-20×Erosion)/2.9+18`；蓝线使用校正 I1 全样本 GAM 曲线。该 CEC 曲线与其他指标历史曲线的模型来源不同，均为描述性预测关系。

Figure data: `data/08_health_response_curves.csv`.
