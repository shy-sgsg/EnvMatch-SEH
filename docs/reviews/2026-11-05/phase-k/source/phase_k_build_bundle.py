from pathlib import Path
import csv, json, re, shutil, sys, subprocess
import pandas as pd
import numpy as np

ROOT = Path.cwd()
RUN = ROOT / 'outputs/exploratory_oracle/phase_k/phase_k_20261104_seed20261104_r1'
SOURCES = {
    'global': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_r2/results/global_response/global_associations.csv',
    'matched': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i2/tables/matched_pair_associations.csv',
    'gam': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_r2/results/nonlinear/full_sample_gam_curves.csv',
    'env_slopes': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/tables/environment_group_slopes_m4.csv',
    'env_interactions': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/tables/environment_interactions_m4.csv',
    'gwr_summary': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i4_r2/tables/gwr_spatial_model_summary.csv',
    'zone_summary': ROOT/'outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2/tables/stress_zone_summary.csv',
    'local_maps': ROOT/'docs/reviews/2026-11-02/phase-j/figures/supplementary/data/U_nine_outcome_local_beta_maps_source.csv',
    'zone_tiles': ROOT/'docs/reviews/2026-11-02/phase-j/figures/data/figure_C_240m_stress_zones.csv',
    'zone_profiles': ROOT/'docs/reviews/2026-11-02/phase-j/figures/data/figure_C_environment_profiles.csv',
    'zone_env_profiles': ROOT/'docs/reviews/2026-11-02/phase-j/spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv',
    'sem_rank': ROOT/'docs/reviews/2026-11-02/phase-j/results/SEM_MODEL_RANKINGS.csv',
    'sem_paths': ROOT/'docs/reviews/2026-11-02/phase-j/results/SEM_PATH_COEFFICIENTS.csv',
    'sem_increment': ROOT/'docs/reviews/2026-11-02/phase-j/results/SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv',
    'sem_env': ROOT/'docs/reviews/2026-11-02/phase-j/results/SEM_ENVIRONMENT_GROUP_COMPARISON.csv',
    'population': ROOT/'docs/reviews/2026-11-02/phase-j/data/population_counts.csv',
}
OUT = RUN/'20yr_only'
for p in [OUT/'tables',OUT/'figures',OUT/'spatial',OUT/'matching',OUT/'sem',RUN/'tables',RUN/'figures',RUN/'provenance']:
    p.mkdir(parents=True,exist_ok=True)

def read(name):
    return pd.read_csv(SOURCES[name])
def rename_outcome(df):
    if 'outcome' in df.columns:
        df = df.copy()
        df['outcome'] = df['outcome'].replace({'SHI_candidate':'SHI'})
    if 'label' in df.columns:
        df = df.copy()
        df['label'] = df['label'].str.replace('土壤健康指数（SHI）','土壤健康指数（SHI）',regex=False)
    return df

def save(df, path):
    df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.8g')

# Exposure-specific source data; the new analysis bundle never carries a 2020 exposure field.
global_df = rename_outcome(read('global'))
global_20 = global_df[(global_df.exposure=='Aa_20yr')&(global_df.population=='mask_all')].copy()
matched_df = rename_outcome(read('matched'))
matched_20 = matched_df[(matched_df.exposure=='Aa_20yr')&(matched_df.population=='all_erosion')].copy()
gam_df = rename_outcome(read('gam'))
gam_20 = gam_df[gam_df.exposure=='Aa_20yr'].copy()
env_df = rename_outcome(read('env_slopes'))
env_20 = env_df[env_df.exposure=='Aa_20yr'].copy()
int_df = rename_outcome(read('env_interactions'))
int_20 = int_df[int_df.exposure=='Aa_20yr'].copy()
local_df = rename_outcome(read('local_maps'))
zone_tiles = rename_outcome(read('zone_tiles'))
zone_profiles = rename_outcome(read('zone_profiles'))
zone_env = rename_outcome(read('zone_env_profiles'))
sem_rank=read('sem_rank'); sem_paths=rename_outcome(read('sem_paths')); sem_inc=read('sem_increment'); sem_env=rename_outcome(read('sem_env'))
# The archived SEM configuration sets E=log1p(Aa_20yr); keep the specific result families used below.
sem_rank20=sem_rank[sem_rank.dataset.isin(['D1_common_env_case','D3_M4_unique_pixels','D4_M4_pair_differences'])].copy()
sem_paths20=sem_paths[sem_paths.dataset.isin(['D1_common_env_case','D3_M4_unique_pixels','D4_M4_pair_differences'])].copy()
sem_inc20=sem_inc[sem_inc.dataset.isin(['D1_common_env_case','D3_M4_unique_pixels','D4_M4_pair_differences'])].copy()
sem_env20=sem_env[sem_env.dataset.str.startswith('D7_')].copy()

for df,path in [
    (global_20,OUT/'tables/global_associations_20yr.csv'),
    (matched_20,OUT/'matching/matched_pair_associations_20yr.csv'),
    (gam_20,OUT/'tables/full_sample_gam_curves_20yr.csv'),
    (env_20,OUT/'matching/environment_group_slopes_20yr.csv'),
    (int_20,OUT/'matching/environment_group_interactions_20yr.csv'),
    (rename_outcome(read('gwr_summary')),OUT/'spatial/gwr_model_summary_20yr.csv'),
    (local_df,OUT/'spatial/nine_outcome_local_beta_maps_20yr.csv'),
    (zone_tiles,OUT/'spatial/stress_zone_tiles_20yr.csv'),
    (zone_profiles,OUT/'spatial/stress_zone_profiles_20yr.csv'),
    (zone_env,OUT/'spatial/stress_zone_environment_profiles_20yr.csv'),
    (sem_rank20,OUT/'sem/sem_model_rankings_20yr.csv'),
    (sem_paths20,OUT/'sem/sem_path_coefficients_20yr.csv'),
    (sem_inc20,OUT/'sem/sem_erosion_increment_20yr.csv'),
    (sem_env20,OUT/'sem/sem_environment_group_paths_20yr.csv'),
]: save(df,path)

labels={'SHI':'土壤健康指数','AN':'碱解氮','OM':'有机质','CEC':'阳离子交换量','pH':'土壤酸碱度','SW':'土壤含水量','BD':'土壤容重','MBC':'微生物量碳','RESP':'土壤呼吸'}
# Explicitly state provenance and selection mode.
config={
 'run_id':'phase_k_20261104_seed20261104_r1','mode':'EXPLORATORY_ORACLE','outcome_guided_selection':True,
 'primary_exposure':'log1p(Aa_20yr)','exposure_scenario':'20yr_only','official_health_inputs':['SHI','AN','OM','CEC','pH','SW','BD','MBC','RESP'],
 'input_policy':'Use supplied SHI and CEC layers as-is; do not reconstruct either layer. Use existing Aa_20yr values; do not rebuild RUSLE products.',
 'independent_single_exposure_policy':'No Aa_2020 variable is used as an exposure, covariate, grouping field, matching field, or model selector in the 20yr_only bundle. Matching memberships and model tables are historical Aa_20yr-specific results.',
 'reuse_policy':'Reuse audited Phase F-J model results and source tables; single-exposure tables are filtered to Aa_20yr or extracted from results explicitly configured with log1p(Aa_20yr). No full model search or GWR/MGWR/SEM refit is performed in this consolidation run.',
 'figures':['A','B','C','D','E','F','G','H','I'],
 'primary_input_tables':{k:str(v.relative_to(ROOT)) for k,v in SOURCES.items()},
 'interpretation':'Exploratory conditional associations from predicted raster values; pixels are spatially dependent and do not equal independent measured sites. Spatial and subgroup patterns are descriptive; no causal claim.'
}
(RUN/'provenance/analysis_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')

# Cross-stage evidence index, retaining the exact population and estimand for each claim.
crosswalk=[
['环境匹配能否保留可比像元？','F','M3/M4空间留出匹配','M4覆盖率中位数0.483；最大绝对SMD中位数0.012','匹配质量好不等于无混杂；探索性设计诊断','docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md'],
['配对内健康差异是否随侵蚀差变化？','F-G','20年侵蚀M4配对差值','SHI/SW/OM差值秩相关约-0.128/-0.192/-0.114（早期约90,848对版本）','与后续111,289对候选版本分开；统计总体不同','docs/reviews/2026-10-27/phase-g/PHASE_G_RESEARCH_REPORT.md'],
['低值筛选会改变结论吗？','H','全样本与结果导向低值筛选','全域保留更完整；CDF筛选后的配对端点相关普遍减弱，空间测试未一致改善','Oracle敏感性，不能把筛选结果当正式阈值或验证','docs/reviews/2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md'],
['九项健康结局的总体方向如何？','I','30m预测像元；完整案例n约384k-388k','20年侵蚀与SW -0.584、SHI -0.469、OM -0.415；CEC +0.168；其余多数为弱至中等负相关','全域秩相关；像元有空间依赖，CEC不作因果解释','outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_r2/results/global_response/global_associations.csv'],
['环境匹配后关联还在吗？','I','M4第二套20年侵蚀匹配，111,289对','SHI/SW/OM配对差值Spearman约-0.144/-0.272/-0.115','差值层相关不等于匹配造成的效应衰减','outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i2/tables/matched_pair_associations.csv'],
['关联是否随位置变化？','I-J','GWR-LCR 7,387个240m单元及770个均衡点模型','SHI/OM/SW局部β中位数约-0.222/-0.395/-0.486；负系数比例约99.6%/100%/100%','局部系数比例不是显著比例；点位来自预测面','docs/reviews/2026-11-02/phase-j/spatial_synthesis/local_coefficient_distribution_summary.csv'],
['哪些环境位置值得进一步核查？','I-J','240m组合单元和候选压力分区','T1约18.882km²、均值Aa=15.67、均值SHI=0.549、β中位数=-0.567','分区经过结果筛选，仅为核查候选，不是正式风险等级','docs/reviews/2026-11-02/phase-j/spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv'],
['控制环境后是否仍有直接关联？','J','D1共同环境完整案例，n=293,145；5km块稳健分段路径','SHI标准化直接β=-0.113；R²=0.601；条件ΔR²=0.0079','结果导向模型比较，非因果中介；SHI来自健康指标体系','docs/reviews/2026-11-02/phase-j/results/SEM_MODEL_RANKINGS.csv'],
['既有PCNM是否已纳入对比？','F-J','现有PCNM结果保留，待专门对照','本阶段未替代或重跑PCNM','跨方法比较仍需后续预设分析与独立复核','docs/RESEARCH_CHARTER.md']]
with (RUN/'tables/phase_fj_evidence_crosswalk.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['研究问题','阶段来源','总体与统计单位','20年侵蚀证据摘要','限制与解读','可追溯来源']);w.writerows(crosswalk)

# Four evidence-led findings.
findings=[
['1','整体上，较高的20年平均侵蚀与较低的SHI、SW、OM同时出现','全域Spearman：SW -0.584（n=388,092），SHI -0.469（n=384,737），OM -0.415（n=388,092）。M4配对差值相关分别约-0.272、-0.144、-0.115（111,289对设计）。','方向在全域和匹配差值中均为负；两种统计单位不能相减来声称匹配削弱幅度。'],
['2','九项结局反应不同，不能说所有土壤指标都一致变差','全域Spearman排序：SW -0.584、SHI -0.469、OM -0.415、RESP -0.272、AN -0.233、BD -0.200、MBC -0.118、pH -0.109、CEC +0.168。','CEC采用供给图层；任何单一指标相关都不能代表整体健康，相关不证明因果。'],
['3','空间局部回归显示负向关系占主导，但强弱因地点而异','GWR-LCR的SHI、OM、SW局部β中位数约-0.222、-0.395、-0.486；负系数单元比例约99.6%、100%、100%。770点GWR/MGWR的系数排序一致性有限。','系数为局部条件估计；负系数占比不是显著性比例，带宽不是精确作用半径。'],
['4','控制核心环境后，20年侵蚀与SHI仍有较弱的负向条件关系','D1完整案例n=293,145，5km块稳健模型：SHI标准化直接β=-0.113，R²=0.601，侵蚀条件ΔR²=0.0079。物理/化学/生物活动维度对应β约-0.389/-0.251/-0.066。','条件关联与预测增量不是因果贡献；空间预测像元不是独立实测点。']]
with (RUN/'tables/four_core_findings.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['序号','发现','定量证据','解释边界']);w.writerows(findings)

# Inventory of the final F-J review artifacts. Data-bearing records get a precise exposure classification.
review_roots=[
 ROOT/'docs/reviews/2026-10-26', ROOT/'docs/reviews/2026-10-27', ROOT/'docs/reviews/2026-10-28',
 ROOT/'docs/reviews/2026-10-29', ROOT/'docs/reviews/2026-10-31/phase-i', ROOT/'docs/reviews/2026-11-02/phase-j']
exts={'.csv','.tsv','.md','.html','.tex','.json','.png','.svg','.pdf','.parquet'}
records=[]
for rr in review_roots:
 if not rr.exists(): continue
 files=sorted(p for p in rr.rglob('*') if p.is_file() and p.suffix.lower() in exts and '.git' not in p.parts and '/source/' not in p.as_posix() and '/data/' not in p.as_posix() and '/inputs/' not in p.as_posix())
 for p in files:
  stage='F' if '2026-10-26' in p.as_posix() else 'G' if '2026-10-27' in p.as_posix() else 'H' if ('2026-10-28' in p.as_posix() or '2026-10-29' in p.as_posix()) else 'I' if '2026-10-31' in p.as_posix() else 'J'
  cls='单独20年侵蚀材料'
  detail='未从文件名判断；请以来源表/配置为准'
  exposure_vals=[]; cols=[]; text=''
  if p.suffix.lower() in {'.csv','.tsv'}:
   try:
    df=pd.read_csv(p,nrows=200000,encoding='utf-8-sig')
    cols=list(df.columns)
    if 'exposure' in df.columns: exposure_vals=sorted(map(str,df['exposure'].dropna().unique()))
    combined=' '.join(cols)
    if {'Aa_20yr','Aa_2020'}.issubset(set(cols)):
     cls='双期同表数据';detail='同一表内含两期侵蚀字段'
    elif 'Aa_20yr' in exposure_vals and 'Aa_2020' in exposure_vals:
     cls='双期同表数据';detail='exposure字段同时含两期数据'
    elif exposure_vals==['Aa_20yr']:
     cls='20年单期数据';detail='exposure字段只含Aa_20yr'
    elif exposure_vals==['Aa_2020']:
     cls='2020单期数据';detail='exposure字段只含Aa_2020'
    elif 'Aa_20yr' in combined and 'Aa_2020' in combined:
     cls='双期变量表';detail='表头同时含两期字段'
   except Exception as e: detail='CSV读取失败：'+type(e).__name__
  elif p.suffix.lower()=='.parquet':
   try:
    import pyarrow.parquet as pq
    cols=pq.read_schema(p).names
    if {'Aa_20yr','Aa_2020'}.issubset(set(cols)): cls='双期输入/分析数据';detail='Parquet schema含两期变量'
   except Exception: pass
  elif p.suffix.lower() in {'.md','.html','.tex','.json'}:
   try:
    text=p.read_text(encoding='utf-8',errors='ignore')
    if 'Aa_20yr' in text and 'Aa_2020' in text:
     cls='涉及双期的说明/配置';detail='正文/配置提及两期；不等同于同表比较'
    elif 'Aa_20yr' in text:
     cls='20年单期说明';detail='正文/配置以20年侵蚀为主'
   except Exception: pass
  elif p.suffix.lower() in {'.png','.svg','.pdf'}:
   # Figure exposure is assigned from its nearest caption/index where possible; otherwise no assumption.
   sidecars=list(p.parent.glob('data/*.csv'))
   stem=p.stem.lower()
   matched_source=None
   for s in sidecars:
    if stem.split('_')[0] in s.stem.lower() or s.stem.lower().startswith(stem.split('_')[0]):
     matched_source=s;break
   if matched_source:
    try:
     sd=pd.read_csv(matched_source,nrows=200000,encoding='utf-8-sig')
     ev=sorted(map(str,sd['exposure'].dropna().unique())) if 'exposure' in sd else []
     if ev==['Aa_20yr','Aa_2020'] or (('Aa_20yr' in sd.columns) and ('Aa_2020' in sd.columns)):
      cls='双期比较图';detail='由同目录双期数据源支持'
     elif ev==['Aa_20yr']:
      cls='20年单期图';detail='由同目录Aa_20yr单期数据源支持'
    except Exception: pass
  records.append([stage,str(p.relative_to(ROOT)),p.suffix.lower(),cls,detail,'outputs/exploratory_oracle/phase_k/phase_k_20261104_seed20261104_r1/20yr_only/'])
with (RUN/'tables/DUAL_EROSION_ARTIFACT_INVENTORY.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['阶段','审查材料路径','文件类型','暴露时期判定','判定依据','20年单期材料位置']);w.writerows(records)

# Core figure source data are copied into the review bundle for direct inspection.
for src,dest in [
 (RUN/'tables/phase_fj_evidence_crosswalk.csv',ROOT/'docs/reviews/2026-11-05/phase-k/results/PHASE_FJ_EVIDENCE_CROSSWALK.csv'),
 (RUN/'tables/DUAL_EROSION_ARTIFACT_INVENTORY.csv',ROOT/'docs/reviews/2026-11-05/phase-k/results/DUAL_EROSION_ARTIFACT_INVENTORY.csv'),
 (RUN/'tables/four_core_findings.csv',ROOT/'docs/reviews/2026-11-05/phase-k/results/FOUR_CORE_FINDINGS.csv')]:
 shutil.copy2(src,dest)

# Figure generation, all figures use only one exposure scenario.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fonts=[f.name for f in font_manager.fontManager.ttflist]
for cand in ['Noto Sans CJK SC','Noto Sans CJK TC','WenQuanYi Micro Hei','AR PL UMing CN']:
 if cand in fonts:
  plt.rcParams['font.sans-serif']=[cand,'DejaVu Sans'];break
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=150
plt.rcParams['savefig.dpi']=220
plt.rcParams['svg.fonttype']='none'
plt.rcParams['axes.titleweight']='bold'
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
C_BLUE='#2F6B91'; C_ORANGE='#D4804B'; C_GREEN='#5D9474'; C_GRAY='#737B83'; C_RED='#B44E52'
figure_files=[]
def done(fig,letter,name):
    if letter=='B':
        fig.subplots_adjust(left=.04,right=.80,wspace=.08,hspace=.20)
    else:
        fig.tight_layout()
    for ext in (['png','svg','pdf'] if '--pdf' in sys.argv else ['png','svg']):
        fig.savefig(OUT/'figures'/f'{letter}_{name}.{ext}',bbox_inches='tight',facecolor='white')
    figure_files.append((letter,name,fig))

# A: full sample, matched pair and nonlinear responses.
fig,axs=plt.subplots(1,3,figsize=(17,5.5))
order=['SW','SHI','OM','RESP','AN','BD','MBC','pH','CEC']
g0=global_20.set_index('outcome').reindex(order);m0=matched_20.set_index('outcome').reindex(order)
y=np.arange(len(order)); h=.36
axs[0].barh(y-h/2,g0.spearman_rho.astype(float),height=h,color=C_BLUE,label='全域像元')
axs[0].barh(y+h/2,m0.spearman_rho.astype(float),height=h,color=C_ORANGE,label='匹配配对差值')
axs[0].set_yticks(y,[labels[k]+'（'+k+'）' for k in order]);axs[0].invert_yaxis();axs[0].axvline(0,color='black',lw=.7)
axs[0].set_xlabel('Spearman 相关系数');axs[0].set_title('总体排序与匹配差值');axs[0].legend(frameon=False,fontsize=8)
ax_shi=axs[1].twinx()
curve_handles=[]
for outcome,color in [('SHI',C_BLUE),('OM',C_GREEN),('SW',C_ORANGE)]:
 d=gam_20[gam_20.outcome==outcome].sort_values('log1p_exposure')
 x=d.log1p_exposure.astype(float).to_numpy(); pred=d.predicted_outcome.astype(float).to_numpy(); lo=d.pointwise_ci95_low.astype(float).to_numpy();hi=d.pointwise_ci95_high.astype(float).to_numpy()
 target_ax=ax_shi if outcome=='SHI' else axs[1]
 line,=target_ax.plot(x,pred,label=labels[outcome],lw=2,color=color);target_ax.fill_between(x,lo,hi,color=color,alpha=.12);curve_handles.append(line)
axs[1].set_xlabel('log(1+20年平均侵蚀)');axs[1].set_ylabel('OM、SW预测值');axs[1].set_title('三项重点结局的非线性曲线');ax_shi.set_ylabel('SHI预测值');ax_shi.set_ylim(.50,.68);ax_shi.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axs[1].legend(curve_handles,[h.get_label() for h in curve_handles],frameon=False,fontsize=8)
# Clear single exposure annotation.
axs[2].axis('off');axs[2].text(.02,.93,'图A读法',fontsize=15,fontweight='bold')
axs[2].text(.02,.77,'左：全域像元相关与匹配差值相关分开展示。\n两者统计单位不同，不能直接相减。',fontsize=11,va='top')
axs[2].text(.02,.54,'中：曲线展示SHI、有机质和土壤含水量\n随20年侵蚀增加时的平均变化趋势。',fontsize=11,va='top')
axs[2].text(.02,.31,'浅色带为95%逐点区间；预测像元存在\n空间依赖，区间不能代替空间独立验证。',fontsize=11,va='top')
fig.suptitle('20年平均实际侵蚀与土壤健康：全域、匹配与非线性证据',fontsize=16)
done(fig,'A','20yr_global_match_nonlinear')

# B: nine-outcome local coefficient maps (20yr Aa source).
plot_outcomes=['SHI','AN','OM','CEC','pH','SW','BD','MBC','RESP']
fig,axs=plt.subplots(3,3,figsize=(13,10))
for ax,o in zip(axs.flat,plot_outcomes):
 d=local_df[local_df.outcome==o]
 sc=ax.scatter(d.x,d.y,c=d.local_beta_standardized,s=2.2,cmap='RdBu_r',vmin=-.75,vmax=.75,linewidths=0,rasterized=True)
 ax.set_title(labels[o]+'（'+o+'）',fontsize=10);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
cax=fig.add_axes([.84,.2,.018,.6]);cb=fig.colorbar(sc,cax=cax);cb.set_label('局部标准化回归系数 β（负值表示同向增加时指标偏低）')
fig.suptitle('九项土壤健康结局的局部回归系数分布（20年平均侵蚀）',fontsize=15)
done(fig,'B','20yr_nine_outcome_local_maps')

# C: selected candidate zones.
fig,axs=plt.subplots(1,2,figsize=(13,5.6))
zone_order=['T1 高Aa—强负向响应','T2 高Aa—弱负向响应','T3 低Aa—低SHI','T4 环境敏感候选','其他/未分类']
zone_colors={'T1 高Aa—强负向响应':'#9D3F45','T2 高Aa—弱负向响应':'#D27B55','T3 低Aa—低SHI':'#5385A3','T4 环境敏感候选':'#A48647','其他/未分类':'#D5D9DD'}
zone_names={'T1 高Aa—强负向响应':'强负向候选区','T2 高Aa—弱负向响应':'高侵蚀对照区','T3 低Aa—低SHI':'低侵蚀低SHI区','T4 环境敏感候选':'环境对照候选区','其他/未分类':'其他位置'}
for z in zone_order:
 d=zone_tiles[zone_tiles.primary_zone==z]
 if len(d): axs[0].scatter(d.x,d.y,s=3,c=zone_colors[z],label=zone_names[z],linewidths=0,rasterized=True)
axs[0].set_aspect('equal');axs[0].set_xticks([]);axs[0].set_yticks([]);axs[0].set_title('240 m空间单元的候选类型');axs[0].legend(frameon=False,fontsize=7,loc='best')
zp=zone_profiles.copy(); zps=zp[zp.zone_code.isin(['T1','T2','T3','T4'])].copy()
xx=np.arange(len(zps)); axs[1].bar(xx-.18,zps.mean_Aa_20yr.astype(float),.36,color=C_ORANGE,label='平均侵蚀')
ax2=axs[1].twinx();ax2.bar(xx+.18,zps.mean_SHI.astype(float),.36,color=C_BLUE,label='平均SHI')
zone_short={'T1':'强负向','T2':'高蚀对照','T3':'低蚀低SHI','T4':'环境对照'}
axs[1].set_xticks(xx,[zone_short[z] for z in zps.zone_code]);axs[1].set_ylabel('平均20年侵蚀');ax2.set_ylabel('平均SHI');axs[1].set_title('各候选区的平均值');axs[1].spines['top'].set_visible(False);ax2.spines['top'].set_visible(False)
axs[1].legend([axs[1].containers[0],ax2.containers[0]],['平均侵蚀','平均SHI'],frameon=False,loc='upper right',fontsize=8)
fig.suptitle('空间敏感位置候选：用于后续核查，不代表正式风险分区',fontsize=15)
done(fig,'C','20yr_candidate_environment_zones')

# D: control hierarchy.
r=sem_rank20[(sem_rank20.dataset=='D1_common_env_case')].copy();r['control']=pd.Categorical(r.control,['M0','M1','M2','M3'],ordered=True);r=r.sort_values('control')
control_names={'M0':'无环境控制','M1':'核心环境控制','M2':'环境主成分控制','M3':'环境加空间趋势'}
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8));xx=np.arange(len(r))
axs[0].plot(xx,r.direct_erosion_beta_to_SHI.astype(float),marker='o',lw=2,color=C_BLUE);axs[0].axhline(0,color='black',lw=.7);axs[0].set_xticks(xx,[control_names[str(c)] for c in r.control]);axs[0].tick_params(axis='x',rotation=20);axs[0].set_ylabel('SHI标准化直接路径 β');axs[0].set_title('侵蚀—SHI条件关联')
axs[1].bar(xx,r.direct_erosion_delta_R2.astype(float),color=C_ORANGE);axs[1].set_xticks(xx,[control_names[str(c)] for c in r.control]);axs[1].tick_params(axis='x',rotation=20);axs[1].set_ylabel('加入侵蚀后的条件ΔR²');axs[1].set_title('侵蚀变量的额外预测信息')
fig.suptitle('不同环境控制层级下的SHI路径');done(fig,'D','20yr_sem_control_hierarchy')

# E: population comparison, same model family.
r=sem_rank20[(sem_rank20.structure=='S2_direct_plus_indirect_SHI')].copy();
order_ds=['D1_common_env_case','D3_M4_unique_pixels','D4_M4_pair_differences']; r=r[r.dataset.isin(order_ds)&r.control.eq('M1')].copy()
lab={'D1_common_env_case':'全域共同样本','D3_M4_unique_pixels':'匹配端点','D4_M4_pair_differences':'匹配配对差值'}
r['名称']=r.dataset.map(lab);r=r.set_index('dataset').reindex(order_ds).dropna(subset=['n'])
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8));xx=np.arange(len(r))
axs[0].bar(xx,r.direct_erosion_beta_to_SHI.astype(float),color=[C_BLUE,C_GREEN,C_ORANGE]);axs[0].axhline(0,color='black',lw=.7);axs[0].set_xticks(xx,r['名称'],rotation=12);axs[0].set_ylabel('SHI标准化直接路径 β');axs[0].set_title('不同总体中的系数')
axs[1].bar(xx,r.n.astype(float),color=[C_BLUE,C_GREEN,C_ORANGE]);axs[1].set_xticks(xx,r['名称'],rotation=12);axs[1].set_ylabel('分析单位数量');axs[1].set_title('各总体大小（单位不同）')
fig.suptitle('全域与匹配总体对照：总体不同，不能视作同一人群的前后比较');done(fig,'E','20yr_full_matched_sem')

# F: paths to dimensions and SHI.
p=sem_paths20[(sem_paths20.dataset=='D1_common_env_case')&(sem_paths20.method=='piecewise_sem')&(sem_paths20.control=='M1')&(sem_paths20['from']=='z_E')].copy()
sel=p[((p.structure=='S1_parallel_dimensions')&p['to'].isin(['physical','chemical','biological_activity']))|((p.structure=='S2_direct_plus_indirect_SHI')&(p['to']=='z_SHI'))].copy()
order_to=['physical','chemical','biological_activity','z_SHI'];disp={'physical':'物理维度','chemical':'化学维度','biological_activity':'生物活动维度','z_SHI':'SHI'}
sel['label']=sel['to'].map(disp);sel['label']=pd.Categorical(sel.label,[disp[x] for x in order_to],ordered=True);sel=sel.sort_values('label')
fig,ax=plt.subplots(figsize=(8.5,4.8)); yy=np.arange(len(sel));b=sel.beta_standardized.astype(float).to_numpy();lo=sel.ci_low.astype(float).to_numpy();hi=sel.ci_high.astype(float).to_numpy()
ax.errorbar(b,yy,xerr=[b-lo,hi-b],fmt='o',color=C_BLUE,capsize=4,lw=1.8);ax.axvline(0,color='black',lw=.8);ax.set_yticks(yy,sel.label.astype(str));ax.invert_yaxis();ax.set_xlabel('标准化路径系数 β（5 km空间块稳健区间）');ax.set_title('20年侵蚀与健康维度/SHI的条件关联');done(fig,'F','20yr_sem_dimension_paths')

# G: conditional increment across controls.
i=sem_inc20[(sem_inc20.dataset=='D1_common_env_case')&(sem_inc20.method=='piecewise_sem')].copy()
i=i[((i.structure=='S1_parallel_dimensions')&i.outcome.isin(['physical','chemical','biological_activity']))|((i.structure=='S2_direct_plus_indirect_SHI')&(i.outcome=='z_SHI'))].copy()
order_ctrl=['M0','M1','M2','M3']; outs=['physical','chemical','biological_activity','z_SHI'];
fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4);width=.18
for j,o in enumerate(outs):
 d=i[i.outcome==o].set_index('control').reindex(order_ctrl)
 ax.bar(x+(j-1.5)*width,d.delta_r2_erosion_conditional.astype(float),width,label=disp[o],color=[C_BLUE,C_ORANGE,C_GREEN,'#7F6AA8'][j])
ax.set_xticks(x,[control_names[z] for z in order_ctrl],rotation=18);ax.set_ylabel('侵蚀变量的条件ΔR²');ax.set_title('逐步加入环境控制后，侵蚀仍提供多少额外预测信息');ax.legend(frameon=False,ncol=2);done(fig,'G','20yr_sem_incremental_fit')

# H: interpretable path diagram based on one declared model.
p=sem_paths20[(sem_paths20.dataset=='D1_common_env_case')&(sem_paths20.method=='piecewise_sem')&(sem_paths20.control=='M1')]
phys=p[(p.structure=='S1_parallel_dimensions')&(p['from']=='z_E')&(p['to'].isin(['physical','chemical','biological_activity']))]
shi=p[(p.structure=='S2_direct_plus_indirect_SHI')]
coef={(r['from'],r['to']):float(r.beta_standardized) for _,r in pd.concat([phys,shi]).iterrows()}
fig,ax=plt.subplots(figsize=(11,5.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
nodes={'erosion':(.12,.50,'20年平均实际侵蚀\n（对数尺度）'),'physical':(.42,.82,'物理维度\nSW、BD'),'chemical':(.42,.59,'化学维度\nOM、AN'),'bio':(.42,.36,'生物活动维度\nMBC、RESP'),'shi':(.83,.59,'土壤健康指数\nSHI')}
for key,(x0,y0,label) in nodes.items():
 ax.text(x0,y0,label,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.55',fc='#F1F6F8' if key!='erosion' else '#F9EDE7',ec='#59798C',lw=1.2))
def arrow(a,b,beta,offset=0):
 x1,y1,_=nodes[a];x2,y2,_=nodes[b];ax.annotate('',xy=(x2-.08,y2),xytext=(x1+.08,y1),arrowprops=dict(arrowstyle='->',lw=1.7,color=C_BLUE if beta<0 else C_ORANGE,connectionstyle=f'arc3,rad={offset}'))
 ax.text((x1+x2)/2,(y1+y2)/2+offset*.1,f'β={beta:.2f}',fontsize=9,ha='center',bbox=dict(fc='white',ec='none',alpha=.8))
for dim,to in [('physical','physical'),('chemical','chemical'),('bio','biological_activity')]:
 val=coef.get(('z_E',to),np.nan);arrow('erosion',dim,val)
for frm,to in [('physical','physical'),('chemical','chemical'),('biological_activity','bio')]:
 # health-dimension to SHI coefficient if present in selected S2.
 rr=shi[(shi['from']==frm)&(shi['to']=='z_SHI')]
 if len(rr):
  key='bio' if frm=='biological_activity' else frm;arrow(key,'shi',float(rr.iloc[0].beta_standardized),offset=.03 if key=='physical' else -.03)
rr=shi[(shi['from']=='z_E')&(shi['to']=='z_SHI')]
if len(rr): arrow('erosion','shi',float(rr.iloc[0].beta_standardized),offset=-.14)
ax.text(.5,.08,'核心环境变量与空间背景作为控制项进入模型；路径表示条件关联，不作因果链解释。',ha='center',fontsize=10,color='#555')
ax.set_title('核心路径结构示意（全域共同样本，控制主要环境背景）',fontsize=14,pad=12);done(fig,'H','20yr_sem_path_diagram')

# I: soil x TWI subgroup comparison.
g=sem_env20[(sem_env20.method=='piecewise_sem')&(sem_env20.structure=='S2_direct_plus_indirect_SHI')&(sem_env20.control=='M1')&(sem_env20['from']=='z_E')&(sem_env20['to']=='z_SHI')].copy()
if len(g):
 g['soil_code']=g['soil_code'].astype(str);g['twi_group']=g['twi_group'].astype(str);g=g.sort_values(['soil_code','twi_group'])
 fig,ax=plt.subplots(figsize=(11,5));yy=np.arange(len(g));b=g.beta_standardized.astype(float).to_numpy();lo=g.ci_low.astype(float).to_numpy();hi=g.ci_high.astype(float).to_numpy()
 colors=[C_BLUE if s=='26' else C_ORANGE if s=='27' else C_GREEN for s in g.soil_code]
 ax.errorbar(b,yy,xerr=[b-lo,hi-b],fmt='none',ecolor='#777',capsize=2,lw=1)
 soil_names={'26':'冲积土','27':'潮土','42':'脱潮土'}
 quartile_names={'q1':'湿度最低四分位','q2':'湿度较低四分位','q3':'湿度较高四分位','q4':'湿度最高四分位'}
 ax.scatter(b,yy,c=colors,s=48,zorder=3);ax.axvline(0,color='black',lw=.8);ax.set_yticks(yy,[f"{soil_names.get(s,s)} / {quartile_names.get(q,q)}" for s,q in zip(g.soil_code,g.twi_group)]);ax.invert_yaxis();ax.set_xlabel('20年平均侵蚀对SHI的标准化路径 β（模型区间）');ax.set_title('土类与地形湿度组合中的关联差异（结果导向候选）');
 from matplotlib.lines import Line2D
 ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=C_BLUE,label='冲积土'),Line2D([0],[0],marker='o',color='w',markerfacecolor=C_ORANGE,label='潮土'),Line2D([0],[0],marker='o',color='w',markerfacecolor=C_GREEN,label='脱潮土')],frameon=False,loc='lower right')
 done(fig,'I','20yr_soil_twi_sem')
else:
 fig,ax=plt.subplots(figsize=(7,4));ax.text(.5,.5,'所选子组结果未返回，需核查数据行。',ha='center');ax.axis('off');done(fig,'I','20yr_soil_twi_sem')

# Copy figure source tables and captions into the review archive.
for src,dst in [
 (OUT/'tables/global_associations_20yr.csv',OUT/'figures/A_source_global.csv'),
 (OUT/'matching/matched_pair_associations_20yr.csv',OUT/'figures/A_source_matched.csv'),
 (OUT/'tables/full_sample_gam_curves_20yr.csv',OUT/'figures/A_source_gam.csv'),
 (OUT/'spatial/nine_outcome_local_beta_maps_20yr.csv',OUT/'figures/B_source_local_betas.csv'),
 (OUT/'spatial/stress_zone_tiles_20yr.csv',OUT/'figures/C_source_zone_tiles.csv'),
 (OUT/'spatial/stress_zone_profiles_20yr.csv',OUT/'figures/C_source_zone_profiles.csv'),
 (OUT/'sem/sem_model_rankings_20yr.csv',OUT/'figures/D_E_source_sem_rankings.csv'),
 (OUT/'sem/sem_path_coefficients_20yr.csv',OUT/'figures/F_H_source_sem_paths.csv'),
 (OUT/'sem/sem_erosion_increment_20yr.csv',OUT/'figures/G_source_sem_increment.csv'),
 (OUT/'sem/sem_environment_group_paths_20yr.csv',OUT/'figures/I_source_sem_groups.csv')]:
 shutil.copy2(src,dst)

# atlas and figure descriptions
figdesc=[
('A','全域相关、匹配差值和GAM响应','展示20年侵蚀与九项结局的全域秩相关、匹配配对差值相关，以及SHI/OM/SW的平滑响应。'),
('B','九项结局的局部回归系数地图','在共同的240 m空间单元上展示GWR-LCR局部标准化系数，负值表示局部条件关联方向为负。'),
('C','空间敏感位置候选分区','展示由20年侵蚀、SHI与局部回归形成的探索性组合区；用于核查，不是正式风险等级。'),
('D','不同环境控制层级下的SHI路径','比较逐步加入环境变量时Aa_20yr—SHI的标准化系数与条件ΔR²。'),
('E','全域和匹配总体的路径对照','展示全域、匹配端点和匹配配对差值的系数及总体量级；三类单位不合并解释。'),
('F','侵蚀与健康维度路径','展示Aa_20yr对物理、化学、生物活动维度及SHI的路径系数和5 km空间块区间。'),
('G','环境控制下的侵蚀预测增量','比较不同控制层级时侵蚀变量对各健康结局的条件预测增量。'),
('H','结构路径图','展示全域共同样本核心模型中的直接关联和健康维度与SHI的路径；不是因果图。'),
('I','土类×地形湿度子组路径','展示土类和湿度分组中的Aa_20yr—SHI系数；分组结果受Oracle筛选影响。')]
html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>20年侵蚀单情景图集</title><style>body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;max-width:1100px;margin:2em auto;line-height:1.6;color:#222}img{width:100%;border:1px solid #ddd}section{margin:2em 0;padding:1em;border-bottom:1px solid #ddd}.note{background:#f3f6f8;padding:1em}</style><h1>20年平均实际侵蚀单情景图集</h1><p class="note">本图集只使用Aa_20yr作为侵蚀暴露变量。全部结果属探索性条件关联；预测栅格不等于独立实测样点，局部系数和候选分区不作因果或正式风险解释。</p>']
for l,n,d in figdesc:
 html.append(f'<section><h2>图{l}：{n}</h2><p>{d}</p><img src="{l}_{n.replace(" ","_")}.png" alt="图{l}"></section>')
# Filenames are actual standardized names.
actual={l:n for l,n,_ in figdesc}
html=[]
html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>20年侵蚀单情景图集</title><style>body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;max-width:1100px;margin:2em auto;line-height:1.6;color:#222}img{width:100%;border:1px solid #ddd}section{margin:2em 0;padding:1em;border-bottom:1px solid #ddd}.note{background:#f3f6f8;padding:1em}</style><h1>20年平均实际侵蚀单情景图集</h1><p class="note">本图集只使用Aa_20yr作为侵蚀暴露变量。全部结果属探索性条件关联；预测栅格不等于独立实测样点，局部系数和候选分区不作因果或正式风险解释。</p>']
for (l,n,d),(l2,fn,_) in zip(figdesc,figure_files):
 html.append(f'<section><h2>图{l}：{n}</h2><p>{d}</p><img src="figures/{fn}.png" alt="图{l}"></section>')
html.append('<p><a href="../../../../../FIGURE_ATLAS.html">全项目去重图集（复用现有项目总图集）</a></p></html>')
(OUT/'20yr_only_figure_atlas.html').write_text('\n'.join(html),encoding='utf-8')

# Figure manifest. PDF is added by --pdf after the artifact marker is invoked.
with (OUT/'figures/FIGURE_MANIFEST.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['图号','文件基名','主题','数据来源','暴露情景','当前格式']);
 for (l,n,d),(l2,fn,_) in zip(figdesc,figure_files):w.writerow([l,fn,n,'Phase F-J已保存数据表的Aa_20yr专用切片','Aa_20yr','PNG/SVG/PDF'])

# Technical notes and run status.
readme='''# Phase K：20年侵蚀单情景汇总\n\n- 运行：`phase_k_20261104_seed20261104_r1`；模式：EXPLORATORY_ORACLE；`outcome_guided_selection=true`。\n- 主暴露：`log1p(Aa_20yr)`。新建的20yr_only材料中不把另一侵蚀时期作为暴露、协变量、分组、匹配或模型筛选字段。匹配成员来自既有Aa_20yr匹配设计。\n- SHI和CEC直接使用现有供给图层，不重算、不版本比较；图题统一分别写SHI、CEC。\n- 本轮集中整理和重绘，复用Phase F-J已保存结果，不重跑全套匹配、低侵蚀筛选、GWR/MGWR或SEM模型搜索。\n- 新图A-I提供PNG、SVG、PDF；图集 `20yr_only_figure_atlas.html`。图源数据在 `figures/*_source_*.csv` 及 `tables/`、`matching/`、`spatial/`、`sem/`。\n- 单期版本是不同暴露情景下的独立报告与图件，不代表独立外部样本或独立验证。\n- 预测栅格像元存在空间依赖，不等于独立实测样点。关联、路径、子组与候选分区均按探索性条件结果解释。\n'''
(OUT/'README.md').write_text(readme,encoding='utf-8')

# Verify bundle source exclusions and record reproducibility metadata.
for p in OUT.rglob('*.csv'):
 header=list(pd.read_csv(p,nrows=0,encoding='utf-8-sig').columns)
 if 'Aa_2020' in header:
  raise RuntimeError(f'Unexpected second exposure field in 20yr-only table: {p}')
(RUN/'RUN_STATUS.md').write_text(f'''# Phase K 运行状态\n\n- 主运行：`phase_k_20261104_seed20261104_r1`。\n- 模式：`EXPLORATORY_ORACLE`；`outcome_guided_selection=true`。\n- 新增20yr_only表、空间数据、匹配汇总、SEM汇总和图件已从已保存的Aa_20yr单期结果表提取。\n- 20yr_only材料排除Aa_2020暴露字段；图集、图表、报告只用Aa_20yr作暴露。\n- 复用Phase F-J结果，未重新搜索匹配参数、子集、GWR/MGWR带宽或SEM模型。\n- 此脚本生成PNG/SVG；PDF需在PDF操作标记之后生成。\n- 状态：表格与图形脚本已生成；正式报告、PDF、固定提交审查材料及推送尚待完成。\n''',encoding='utf-8')
print('bundle tables and PNG/SVG generated:',len(figure_files),'figures; inventory rows:',len(records))
if '--pdf' in sys.argv:
    report_dir=OUT/'reports'
    report_md=report_dir/'正式学术研究报告_20年侵蚀单情景.md'
    report_pdf=report_dir/'正式学术研究报告_20年侵蚀单情景.pdf'
    cmd=['pandoc',report_md.name,'--from','markdown','--to','pdf','--standalone','--number-sections','--toc',
         '--pdf-engine=xelatex','--resource-path=.:..:../figures','-V','CJKmainfont=Noto Sans CJK SC',
         '-V','mainfont=Noto Sans CJK SC','-V','geometry:margin=2.2cm','-V','fontsize=10.5pt','-o',report_pdf.name]
    subprocess.run(cmd,cwd=report_dir,check=True)
    print('academic PDF generated:',report_pdf)
