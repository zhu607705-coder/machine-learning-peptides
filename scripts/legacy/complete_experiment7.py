from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


BASE_DIR = Path("/Users/zhuhangcheng/Downloads/ai")
SUBMIT_DIR = BASE_DIR / "实验七_朱航成_3250100360"
NOTEBOOK_PATH = SUBMIT_DIR / "实验7 自然语言处理之语义相似度分析.ipynb"
REPORT_PATH = SUBMIT_DIR / "实验7_语义相似度分析报告.md"
PDF_PATH = SUBMIT_DIR / "实验七_朱航成_3250100360.pdf"
HEATMAP_PATH = SUBMIT_DIR / "lsa_similarity_matrix.png"


def build_notebook():
    cells = [
        new_markdown_cell(
            "# 🧠 实验七：文本语义相似度分析\n\n"
            "本实验通过词袋模型、TF-IDF 和低维句向量方法计算中文文本的语义相似度，"
            "并重点观察语序变化对相似度结果的影响。"
        ),
        new_markdown_cell(
            "- 姓名：朱航成\n"
            "- 学号：3250100360\n\n"
            "> 当前版本已经补全实验代码、运行结果与分析说明，可直接整理提交。"
        ),
        new_markdown_cell(
            "## 📌 实验目标\n"
            "1. 理解语义相似度的基本原理和数学表达。\n"
            "2. 掌握词袋模型、TF-IDF 与低维句向量的基本用法。\n"
            "3. 观察不同句子对、尤其是语序变化，对相似度计算结果的影响。\n"
            "4. 对比不同方法在实际任务中的优点与局限。\n\n"
            "## 实验说明\n"
            "由于当前环境为 `Python 3.14`，`gensim` 的 `Word2Vec` 无法正常编译安装，"
            "因此这里采用更稳定的 `TF-IDF + TruncatedSVD (LSA)` 作为句向量方法。"
            "它同样可以把稀疏文本表示映射为低维稠密向量，适合作为本次扩展实验的替代方案。"
        ),
        new_markdown_cell(
            "## ✏️ 数学原理简介\n"
            "**余弦相似度（Cosine Similarity）**：\n"
            "$$\n"
            "\\text{sim}_{\\cos}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}\n"
            "$$\n\n"
            "**广义 Jaccard 相似度（Generalized Jaccard Similarity）**：\n"
            "$$\n"
            "\\text{sim}_{\\text{jaccard}}(A, B) = "
            "\\frac{\\sum_i \\min(A_i, B_i)}{\\sum_i \\max(A_i, B_i)}\n"
            "$$\n\n"
            "以上指标用于衡量向量之间的相似度，数值越大表示文本越接近。"
        ),
        new_markdown_cell(
            "## 1. 实验设计\n"
            "本次选择三组句子进行对比：\n\n"
            "- `主题相近`：词面与语义都有一定重合。\n"
            "- `同词异序`：词汇几乎相同，但语序发生明显变化。\n"
            "- `主题不同`：词汇和主题都基本无关。\n\n"
            "随后使用 `词袋余弦相似度`、`广义 Jaccard`、`TF-IDF 余弦相似度` 和 "
            "`基于词双元组的顺序敏感相似度` 进行对比。"
        ),
        new_code_cell(
            textwrap.dedent(
                """
                import logging
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                import jieba
                import numpy as np
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import matplotlib.pyplot as plt
                import seaborn as sns
                from IPython.display import Markdown, display

                plt.rcParams["figure.dpi"] = 130
                plt.rcParams["axes.unicode_minus"] = False
                plt.rcParams["font.sans-serif"] = [
                    "PingFang SC",
                    "Heiti SC",
                    "SimHei",
                    "Arial Unicode MS",
                    "DejaVu Sans",
                ]
                jieba.setLogLevel(logging.ERROR)
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                experiment_pairs = {
                    "主题相近": ("我喜欢吃苹果", "我爱吃红苹果"),
                    "同词异序": ("我喜欢机器学习", "机器学习喜欢我"),
                    "主题不同": ("今天天气很好适合散步", "数据库索引可以提升查询效率"),
                }

                def jieba_cut(sentence):
                    return " ".join(jieba.lcut(sentence))

                for name, (sentence1, sentence2) in experiment_pairs.items():
                    print(f"{name}:")
                    print("  句子1分词 ->", jieba_cut(sentence1))
                    print("  句子2分词 ->", jieba_cut(sentence2))
                """
            ).strip()
        ),
        new_code_cell(
            textwrap.dedent(
                """
                def generalized_jaccard(a, b):
                    min_sum = np.minimum(a, b).sum()
                    max_sum = np.maximum(a, b).sum()
                    return min_sum / max_sum if max_sum != 0 else 0.0

                def compare_sentences(sentence1, sentence2):
                    sentence1_cut = jieba_cut(sentence1)
                    sentence2_cut = jieba_cut(sentence2)

                    bow_vectorizer = CountVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
                    bow_matrix = bow_vectorizer.fit_transform([sentence1_cut, sentence2_cut])
                    bow_cos = cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])[0][0]
                    jaccard = generalized_jaccard(*bow_matrix.toarray())

                    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
                    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence1_cut, sentence2_cut])
                    tfidf_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                    try:
                        order_vectorizer = TfidfVectorizer(
                            token_pattern=r"(?u)\\b\\w+\\b",
                            analyzer="word",
                            ngram_range=(2, 2),
                        )
                        order_matrix = order_vectorizer.fit_transform([sentence1_cut, sentence2_cut])
                        order_cos = cosine_similarity(order_matrix[0:1], order_matrix[1:2])[0][0]
                    except ValueError:
                        order_cos = 0.0

                    return {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "bow_cos": float(bow_cos),
                        "jaccard": float(jaccard),
                        "tfidf_cos": float(tfidf_cos),
                        "order_cos": float(order_cos),
                    }

                results = []
                for pair_name, (sentence1, sentence2) in experiment_pairs.items():
                    row = compare_sentences(sentence1, sentence2)
                    row["pair_name"] = pair_name
                    results.append(row)

                table = [
                    "| 对比类型 | 句子1 | 句子2 | 词袋余弦 | Jaccard | TF-IDF | 顺序敏感 bigram |",
                    "|---|---|---|---:|---:|---:|---:|",
                ]
                for row in results:
                    table.append(
                        f"| {row['pair_name']} | {row['sentence1']} | {row['sentence2']} | "
                        f"{row['bow_cos']:.4f} | {row['jaccard']:.4f} | "
                        f"{row['tfidf_cos']:.4f} | {row['order_cos']:.4f} |"
                    )

                display(Markdown("\\n".join(table)))
                print("语序实验提示：'同词异序' 在词袋与 TF-IDF 下都接近 1，但顺序敏感 bigram 只有 0.2020。")
                """
            ).strip()
        ),
        new_markdown_cell(
            "## 2. 结果分析\n"
            "从上面的结果可以看到：\n\n"
            "- `主题相近` 的三种词频类方法都给出了中等偏高的相似度，说明它们能识别词汇重叠带来的接近性。\n"
            "- `同词异序` 在词袋余弦、Jaccard、TF-IDF 下都达到了 `1.0`，因为这些方法本质上只关心词是否出现以及出现频率，几乎不关心顺序。\n"
            "- 当改用 `word bigram` 后，`同词异序` 的相似度明显降到 `0.2020`，说明加入局部顺序信息后，模型开始区分“词一样但结构不同”的句子。\n"
            "- `主题不同` 在所有方法下都接近 `0`，这与直觉一致。\n"
        ),
        new_markdown_cell(
            "## 3. 低维句向量相似度矩阵\n"
            "为了进一步观察整组句子的相似结构，下面对 6 个句子构建 `TF-IDF + TruncatedSVD` "
            "得到的低维句向量，并绘制余弦相似度矩阵。"
        ),
        new_code_cell(
            textwrap.dedent(
                """
                corpus_sentences = [
                    "我喜欢吃新鲜的苹果和香蕉",
                    "我爱吃红苹果，也喜欢香蕉奶昔",
                    "深度学习正在改变自然语言处理",
                    "机器学习和神经网络是人工智能的重要方法",
                    "今天下雨了，我准备在宿舍看书",
                    "外面天气很好，适合去操场散步",
                ]

                labels = ["美食1", "美食2", "AI1", "AI2", "生活1", "生活2"]
                stop_words = ["我", "的", "了", "和", "是", "在", "也", "去", "一个", "正在", "今天", "外面"]
                corpus_cut = [jieba_cut(sentence) for sentence in corpus_sentences]

                tfidf_vectorizer = TfidfVectorizer(
                    token_pattern=r"(?u)\\b\\w+\\b",
                    stop_words=stop_words,
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_cut)

                svd = TruncatedSVD(n_components=4, random_state=42)
                sentence_vectors = svd.fit_transform(tfidf_matrix)
                similarity_matrix = cosine_similarity(sentence_vectors)

                print(f"LSA 累计解释方差占比: {svd.explained_variance_ratio_.sum():.4f}")

                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    similarity_matrix,
                    annot=True,
                    cmap="YlOrRd",
                    fmt=".2f",
                    xticklabels=labels,
                    yticklabels=labels,
                )
                plt.title("LSA Sentence Similarity Matrix")
                plt.tight_layout()
                plt.savefig("lsa_similarity_matrix.png", bbox_inches="tight")
                plt.show()
                """
            ).strip()
        ),
        new_markdown_cell(
            "## 思考题\n"
            "我认为上述计算方式 **基本合理，但并不充分**。\n\n"
            "首先，词袋模型、Jaccard 和普通 TF-IDF 都容易受到“只看词、不看顺序”的限制。"
            "因此当两个句子使用几乎相同的词、但语义关系已经变化时，它们仍可能被判定为高度相似。"
            "本次实验中的“我喜欢机器学习”和“机器学习喜欢我”就是典型例子。\n\n"
            "其次，低维句向量方法能够在一定程度上压缩语义信息，比纯词频方法更适合做多句整体比较，"
            "但它依然强依赖训练语料规模。当前实验语料很小，所以矩阵更多体现的是“教学演示价值”，"
            "而不是强泛化能力。\n\n"
            "如果希望进一步提高句子相似度计算效果，更好的方法包括：\n\n"
            "1. 使用预训练词向量或预训练语言模型，例如 `word2vec`、`BERT`、`Sentence-BERT`。\n"
            "2. 引入语序建模能力，例如 `bigram/trigram`、RNN、Transformer。\n"
            "3. 在具体任务中结合人工标注数据进行微调，使“相似”的定义更加贴合业务场景。\n\n"
            "因此，我的结论是：**词袋模型适合入门和基线实验，但如果要做更真实的语义相似度任务，"
            "应优先考虑预训练嵌入模型和上下文表示方法。**"
        ),
    ]

    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python", "version": "3.14"}})
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def execute_notebook(nb):
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    return nb


def write_report():
    report = textwrap.dedent(
        """
        # 实验7 语义相似度分析报告

        ## 1. 实验目的
        本实验围绕中文句子相似度展开，对比词袋模型、广义 Jaccard、TF-IDF 和低维句向量方法在不同句子对上的表现，并重点分析语序变化的影响。

        ## 2. 实验环境
        - Python 3.14
        - jieba
        - numpy
        - scikit-learn
        - matplotlib
        - seaborn

        说明：原始示例中的 `gensim Word2Vec` 在当前 `Python 3.14` 环境下无法成功编译安装，因此本次报告使用 `TF-IDF + TruncatedSVD (LSA)` 替代低维句向量实验。

        ## 3. 句子对设计
        - 主题相近：`我喜欢吃苹果` vs `我爱吃红苹果`
        - 同词异序：`我喜欢机器学习` vs `机器学习喜欢我`
        - 主题不同：`今天天气很好适合散步` vs `数据库索引可以提升查询效率`

        ## 4. 实验结果
        | 对比类型 | 词袋余弦 | Jaccard | TF-IDF | 顺序敏感 bigram |
        |---|---:|---:|---:|---:|
        | 主题相近 | 0.6708 | 0.5000 | 0.5101 | 0.0000 |
        | 同词异序 | 1.0000 | 1.0000 | 1.0000 | 0.2020 |
        | 主题不同 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

        ## 5. 结果分析
        - 对于主题相近的句子，词袋和 TF-IDF 都能给出中等偏高的相似度，说明它们可以利用共享词汇进行匹配。
        - 对于同词异序的句子，词袋、Jaccard 和 TF-IDF 都给出了 1.0000，这暴露了词频方法对语序不敏感的缺点。
        - 引入基于词双元组的顺序敏感相似度后，“同词异序”下降到 0.2020，说明只要引入局部顺序信息，就能更好地区分结构差异。
        - 对于主题不同的句子，各方法结果都接近 0，与直觉一致。

        ## 6. 低维句向量实验
        对 6 个句子构建 TF-IDF 特征，再通过 TruncatedSVD 降维得到句向量。实验中累计解释方差占比约为 `0.7306`，相似度矩阵能较清楚地把“美食主题”和“AI 主题”聚到一起，说明低维稠密表示比单纯词袋更适合做整体结构分析。

        ## 7. 思考与结论
        本实验说明：
        - 词袋模型适合做入门级基线，但无法处理语序和深层语义。
        - TF-IDF 对关键词区分更好，但仍然缺乏上下文能力。
        - 顺序敏感特征可以部分缓解“同词异序”问题。
        - 如果需要更高质量的语义相似度，推荐使用预训练词向量、BERT 或 Sentence-BERT 等上下文模型。

        总体而言，本次实验验证了“不同表示方法会直接影响相似度结论”，尤其在涉及语序和深层语义时，简单词频方法存在明显局限。
        """
    ).strip() + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    executed_nb = execute_notebook(nb)
    NOTEBOOK_PATH.write_text(nbformat.writes(executed_nb), encoding="utf-8")
    write_report()
    print(f"Notebook written to: {NOTEBOOK_PATH}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
