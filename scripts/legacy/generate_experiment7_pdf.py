from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = Path("/Users/zhuhangcheng/Downloads/ai/实验七_朱航成_3250100360")
PDF_PATH = BASE_DIR / "实验七_朱航成_3250100360.pdf"
HEATMAP_PATH = BASE_DIR / "lsa_similarity_matrix.png"


def build_pdf():
    registerFont(UnicodeCIDFont("STSong-Light"))

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="STSong-Light",
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "HeadingCN",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#222222"),
        spaceBefore=6,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )

    story = [
        Paragraph("实验七 朱航成 3250100360", title_style),
        Paragraph("文本语义相似度分析实验报告", title_style),
        Spacer(1, 0.2 * cm),
        Paragraph("一、实验目的", heading_style),
        Paragraph(
            "本实验围绕中文句子相似度展开，对比词袋模型、广义 Jaccard、TF-IDF 和低维句向量方法在不同句子对上的表现，并重点分析语序变化的影响。",
            body_style,
        ),
        Paragraph("二、实验环境", heading_style),
        Paragraph(
            "Python 3.14，jieba，numpy，scikit-learn，matplotlib，seaborn。",
            body_style,
        ),
        Paragraph(
            "说明：原始示例中的 gensim Word2Vec 在当前 Python 3.14 环境下无法成功编译安装，因此本次实验使用 TF-IDF + TruncatedSVD（LSA）替代低维句向量实验。",
            body_style,
        ),
        Paragraph("三、句子对设计", heading_style),
        Paragraph("1. 主题相近：我喜欢吃苹果 vs 我爱吃红苹果", body_style),
        Paragraph("2. 同词异序：我喜欢机器学习 vs 机器学习喜欢我", body_style),
        Paragraph("3. 主题不同：今天天气很好适合散步 vs 数据库索引可以提升查询效率", body_style),
        Paragraph("四、实验结果", heading_style),
    ]

    table_data = [
        ["对比类型", "词袋余弦", "Jaccard", "TF-IDF", "顺序敏感 bigram"],
        ["主题相近", "0.6708", "0.5000", "0.5101", "0.0000"],
        ["同词异序", "1.0000", "1.0000", "1.0000", "0.2020"],
        ["主题不同", "0.0000", "0.0000", "0.0000", "0.0000"],
    ]
    table = Table(table_data, colWidths=[3.0 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 4.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#666666")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))
    story.extend(
        [
            Paragraph("五、结果分析", heading_style),
            Paragraph("1. 主题相近的句子在词袋和 TF-IDF 下都表现出中等偏高的相似度，说明这些方法可以利用共享词汇完成基础匹配。", body_style),
            Paragraph("2. 同词异序的句子在词袋、Jaccard 和 TF-IDF 下都达到 1.0000，说明这类词频方法几乎不考虑语序。", body_style),
            Paragraph("3. 引入基于词双元组的顺序敏感特征后，相似度下降到 0.2020，说明局部顺序信息可以有效区分结构差异。", body_style),
            Paragraph("4. 主题完全不同的句子在各方法下都接近 0，结果符合直觉。", body_style),
            Paragraph("六、低维句向量实验", heading_style),
            Paragraph("对 6 个句子构建 TF-IDF 特征后，再使用 TruncatedSVD 降维为低维句向量。实验中的累计解释方差占比为 0.7306，相似度矩阵能把美食主题和 AI 主题较好地区分出来。", body_style),
        ]
    )

    if HEATMAP_PATH.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(HEATMAP_PATH), width=14.8 * cm, height=10.8 * cm))
        story.append(Spacer(1, 0.2 * cm))

    story.extend(
        [
            Paragraph("七、思考与结论", heading_style),
            Paragraph("词袋模型适合作为入门基线，但无法刻画语序和深层语义；TF-IDF 能增强关键词区分能力，但本质上仍是词频模型。若需要更高质量的句子相似度判断，应优先考虑预训练词向量、BERT 或 Sentence-BERT 等上下文方法。", body_style),
            Paragraph("本次实验验证了一个核心结论：文本表示方式会直接影响相似度结论，尤其当句子之间存在语序变化时，简单词频方法会产生明显偏差。", body_style),
        ]
    )

    doc.build(story)
    print(f"PDF written to: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
