# SEM模型选择：四类总体各自报告四个目标

AIC候选使用同一总体、SHI结果和成熟lavaan估计器比较S2/S3；侵蚀增量、负向路径和解释候选使用同一总体的S2分段路径模型。排名不跨总体合并。所有结果来自Oracle探索，属于表观、结果导向比较。

## D1_common_env_case｜A｜整体拟合候选

总体：全域共同完整像元。候选模型：CB_SEM_lavaan_ML，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一总体SHI方程、S2/S3候选中的最低AIC；数值=1732266.6868。

需要结合自由度判断；df=0为饱和模型，其CFI/RMSEA不能作为独立拟合质量证据。 CFI=1.000; TLI=1.000; RMSEA=0.000; SRMR=5.54e-10; df=0.0; n=293145

## D1_common_env_case｜B｜侵蚀增量最大

总体：全域共同完整像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M0。判据：同一SHI方程中最大条件ΔR²；数值=0.0170。

这是模型内侵蚀的条件预测增量；不据此将未调整模型解释为最佳科学模型。 R²full=0.553; R²without=0.536; β=-0.157; n=293145

## D1_common_env_case｜C｜负向直接路径最明显

总体：全域共同完整像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M0。判据：同一总体S2分段路径中侵蚀→SHI标准化β最小；数值=-0.1568。

路径绝对值大不代表因果效应或外部预测最好。 SHI R²=0.553; n=293145

## D1_common_env_case｜D｜最佳科学解释候选

总体：全域共同完整像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M1。判据：核心环境控制 + 5 km空间块协方差路径区间；数值=-0.1130。

保留可解释环境控制和空间块协方差；不称为拟合冠军。 SHI R²=0.601; 侵蚀ΔR²=0.0079; n=293145

## D2_240m_blocks｜A｜整体拟合候选

总体：240米空间汇总单元。候选模型：CB_SEM_lavaan_ML，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一总体SHI方程、S2/S3候选中的最低AIC；数值=7145.7563。

需要结合自由度判断；df=0为饱和模型，其CFI/RMSEA不能作为独立拟合质量证据。 CFI=1.000; TLI=1.000; RMSEA=0.000; SRMR=2.95e-09; df=0.0; n=7387

## D2_240m_blocks｜B｜侵蚀增量最大

总体：240米空间汇总单元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M0。判据：同一SHI方程中最大条件ΔR²；数值=0.0046。

这是模型内侵蚀的条件预测增量；不据此将未调整模型解释为最佳科学模型。 R²full=0.722; R²without=0.718; β=-0.097; n=7375

## D2_240m_blocks｜C｜负向直接路径最明显

总体：240米空间汇总单元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M0。判据：同一总体S2分段路径中侵蚀→SHI标准化β最小；数值=-0.0973。

路径绝对值大不代表因果效应或外部预测最好。 SHI R²=0.722; n=7375

## D2_240m_blocks｜D｜最佳科学解释候选

总体：240米空间汇总单元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M1。判据：核心环境控制 + 5 km空间块协方差路径区间；数值=-0.0611。

保留可解释环境控制和空间块协方差；不称为拟合冠军。 SHI R²=0.768; 侵蚀ΔR²=0.0016; n=7375

## D3_M4_unique_pixels｜A｜整体拟合候选

总体：M4匹配端点像元。候选模型：CB_SEM_lavaan_ML，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一总体SHI方程、S2/S3候选中的最低AIC；数值=843200.1292。

需要结合自由度判断；df=0为饱和模型，其CFI/RMSEA不能作为独立拟合质量证据。 CFI=1.000; TLI=1.000; RMSEA=0.000; SRMR=9.14e-09; df=0.0; n=142451

## D3_M4_unique_pixels｜B｜侵蚀增量最大

总体：M4匹配端点像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一SHI方程中最大条件ΔR²；数值=0.0150。

这是模型内侵蚀的条件预测增量；不据此将未调整模型解释为最佳科学模型。 R²full=0.620; R²without=0.605; β=-0.152; n=142256

## D3_M4_unique_pixels｜C｜负向直接路径最明显

总体：M4匹配端点像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一总体S2分段路径中侵蚀→SHI标准化β最小；数值=-0.1516。

路径绝对值大不代表因果效应或外部预测最好。 SHI R²=0.620; n=142256

## D3_M4_unique_pixels｜D｜最佳科学解释候选

总体：M4匹配端点像元。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M1。判据：核心环境控制 + 5 km空间块协方差路径区间；数值=-0.1413。

保留可解释环境控制和空间块协方差；不称为拟合冠军。 SHI R²=0.562; 侵蚀ΔR²=0.0132; n=142256

## D4_M4_pair_differences｜A｜整体拟合候选

总体：M4配对差值。候选模型：CB_SEM_lavaan_ML，结构=S2_direct_plus_indirect_SHI，控制=M2。判据：同一总体SHI方程、S2/S3候选中的最低AIC；数值=779739.6136。

需要结合自由度判断；df=0为饱和模型，其CFI/RMSEA不能作为独立拟合质量证据。 CFI=1.000; TLI=1.000; RMSEA=0.000; SRMR=2.14e-09; df=0.0; n=110656

## D4_M4_pair_differences｜B｜侵蚀增量最大

总体：M4配对差值。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一SHI方程中最大条件ΔR²；数值=0.0054。

这是模型内侵蚀的条件预测增量；不据此将未调整模型解释为最佳科学模型。 R²full=0.322; R²without=0.317; β=-0.082; n=110656

## D4_M4_pair_differences｜C｜负向直接路径最明显

总体：M4配对差值。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M3。判据：同一总体S2分段路径中侵蚀→SHI标准化β最小；数值=-0.0821。

路径绝对值大不代表因果效应或外部预测最好。 SHI R²=0.322; n=110656

## D4_M4_pair_differences｜D｜最佳科学解释候选

总体：M4配对差值。候选模型：piecewise_sem，结构=S2_direct_plus_indirect_SHI，控制=M1。判据：核心环境控制 + 5 km空间块协方差路径区间；数值=-0.0688。

保留可解释环境控制和空间块协方差；不称为拟合冠军。 SHI R²=0.308; 侵蚀ΔR²=0.0039; n=110656
