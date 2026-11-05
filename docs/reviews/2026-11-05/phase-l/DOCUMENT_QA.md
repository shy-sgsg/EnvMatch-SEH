# 文档生成与排版核对

- 合并报告采用同一份Markdown正文生成LaTeX、PDF和DOCX；封面题名为《土壤侵蚀对土壤健康的响应特征、环境异质性及作用路径研究——以河南省温县为例》，正文严格分为两章。
- PDF由XeLaTeX生成，A4纸，共43页；本次更新后将43页渲染为缩略接触表，检查标题页、目录、正文、表格、图片、页眉页脚和页面空白，没有发现明显截断或空白页。
- DOCX已读取并核对OOXML结构：A4页面、约2.1厘米页边距、15张表、20个内嵌图片，且包含指定的两章标题。当前环境缺少`pdf2image`与LibreOffice/soffice，无法进行DOCX页面级目视渲染；需在Word或LibreOffice打开确认分页。
- 报告包含20幅不重复图和15张表；所引用图像路径均存在。CEC在报告中统一写作“CEC”，未出现旧CEC公式或“校正CEC”字样。
- 交付文件：`FINAL_ACADEMIC_RESEARCH_REPORT_30000PLUS.md`、`.docx`、`.pdf`；根目录便捷副本为`docs/PROJECT_FULL_REPORT.*`。
