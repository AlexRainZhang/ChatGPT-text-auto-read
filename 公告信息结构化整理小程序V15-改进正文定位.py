""" 
版本号：公告信息结构化整理小程序V15

用法：选择一个储存来自wind的上市公司PDF公告的文件夹（必须是wind的文件命名形式），运行该程序，以完成对PDF的结构化分析工作。
"""

# 导入必要的库
import tkinter as tk
from tkinter import filedialog
import PyPDF2
import re
import pandas as pd
from pathlib import Path


# 函数：将中文数字转换为阿拉伯数字
def chinese_to_arabic(cn_num):
    """
    将中文数字转换为阿拉伯数字。
    
    参数:
        cn_num (str): 中文数字字符串。
        
    返回:
        int: 转换后的阿拉伯数字。
    """
    # 定义中文数字和阿拉伯数字之间的映射
    numerals = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
                '六': 6, '七': 7, '八': 8, '九': 9}
    # 定义权重映射，用于处理“十”、“百”、“千”等
    weights = {'十': 10, '百': 100, '千': 1000}
    
    # 处理特例"十"
    if cn_num == "十":
        return 10

    # 初始化总数和当前权重单位
    num = 0
    multiplier = 0
    for char in cn_num:
        if char in numerals:
            multiplier = numerals[char]
        elif char in weights:
            if multiplier == 0:
                multiplier = 1
            num += multiplier * weights[char]
            multiplier = 0

    # If a numeral is the last character, add it to the number
    num += multiplier
    return num

# 函数：定义可能的章节模式
def determine_chapter_pattern(content):
    """
    这里定义了多个正则表达式模式，是因为不同的文档可能使用不同的章节编号格式。
    通过这些模式，我们能够更准确地匹配不同文档中的章节。
    
    参数:
        content (str): 待扫描的文本内容。
        
    返回:
        re.Pattern: 检测到的章节的正则表达式模式。
    """
    # 定义可能的章节模式
    numerals_pattern = r"[零一二三四五六七八九十百]+"
    chapter_patterns = [
        re.compile(fr"(第\s*{numerals_pattern}\s*章)\s*([^\n.]+)"),   #第零一二三四五六七八九十百章
        re.compile(fr"(第\s*{numerals_pattern}\s*节)\s*([^\n.]+)"),   #第零一二三四五六七八九十百节
        re.compile(fr"({numerals_pattern}\s*、)\s*([^\n.]+)"),     #零一二三四五六七八九十百、
        re.compile(fr"(（\s*{numerals_pattern}\s*）)\s*([^\n.]+)"),   #（零一二三四五六七八九十百）
        re.compile(fr"(第\s*{numerals_pattern}\s*条)\s*([^\n.]+)")    #第零一二三四五六七八九十百条
    ]
    # 查找匹配的模式
    pattern_matches = {}
    for pattern in chapter_patterns:
        matches = pattern.findall(content)
        if matches:
            pattern_matches[pattern] = matches

    # 如果只有一个匹配的模式，直接返回
    if len(pattern_matches) == 1:
        return list(pattern_matches.keys())[0]
    
    # 找到重复内容超过4次及以上的模式，并移除（对应多个子标题的情况），这是一个不鲁棒的结构
    for pattern, matches in list(pattern_matches.items()):
        content_counts = {}
        for content, _ in matches:
            content_counts[content] = content_counts.get(content, 0) + 1
        for _, count in content_counts.items():
            if count > 3:
                del pattern_matches[pattern]
                break
    
    # 选择最佳的模式
    if pattern_matches:
        return max(pattern_matches, key=lambda k: len(pattern_matches[k]))
    
    return None

# 函数：确定目录的起点和推测终点
def determine_toc_bounds(document_content):
    """
    确定目录的起始和结束位置是为了后续更准确地从文档中提取章节信息。
    如果文档没有目录，toc_start会被设置为0。
    
    参数:
        document_content (str): 文档的全文内容。
        
    返回:
        tuple: “目录”的起始位置、结束位置以及正文的起始位置。
    """
    # 检查文档是否包含“目录”
    has_toc = re.search(r"目\s*录", document_content)
    
    # 记录是否存在“目录”
    if has_toc:
        toc_exist = True
    else:
        toc_exist = False    

    # 定位目录起点，如果没有目录，toc_start 设置为 0
    toc_start = re.search(r"目\s*录", document_content).start() if has_toc else 0

    # 给 toc_end_guess 设定一个无穷大的初始值
    toc_end_guess = len(document_content)
        
    return toc_start, toc_end_guess,toc_exist

# 函数：推测正文的起点
def determine_content_start(document_content, first_chapter_title, toc_start, toc_exist):
    """
    正文的起点用于后续的章节信息提取。
    我们首先查找第一个有效章节标题在文档中的位置，然后将其视为正文的起点。
    
    参数:
        document_content (str): 文档的全文内容。
        first_chapter_title (str): 第一个有效的章节标题。
        toc_start (int): 目录的结束位置。
        toc_exist (bool): 指示文档是否包含“目录”。
        
    返回:
        int: 正文的开始位置。
    """
   
    # 构建用于查找 first_chapter_title 的正则表达式
    search_title_pattern = re.compile(re.escape(first_chapter_title).replace(r'\\ ', r'\s*'))
    
    # 查找所有匹配项的位置
    matches = [m for m in search_title_pattern.finditer(document_content, toc_start)]

    if len(matches) > 1:
        # 通常，第二个匹配项应该是正文中的 search_title
        content_start = matches[1].start()
    elif matches:
        # 只找到一个匹配项，很可能是目录中的
        content_start = matches[0].start()
    else:
        # 没有找到匹配项，保持 content_start 不变
        content_start = toc_start

    # 将 content_start 向前推 20 个字符
    content_start = max(content_start - 20, 0)
    
    return content_start

# 函数：查找章节编号的位置
def find_chapter(document_content, chapter_number, content_start):
    """
    在文档内容中查找给定章节编号的位置。
    
    参数:
        document_content (str): 文档的全文内容。
        chapter_number (str): 需要查找的章节编号（如 '第 一 章'）。
        content_start (int): 正文的开始位置。
        
    返回:
        int: 给定章节编号在文档中的开始位置。
    """
    # 构建用于查找 chapter_number 的正则表达式
    pattern_str = chapter_number.replace(" ", r"\s*")
    pattern = re.compile(pattern_str)
    
    # 查找匹配项
    match = pattern.search(document_content, content_start)
    
    if match:
        return match.start()
    else:
        return None

# 函数：将PDF文件转化为同名TXT文件
def convert_pdf_to_txt(pdf_path, txt_path):
    """
    将指定路径的PDF文件转化为TXT文件后存储在指定路径。
    
    参数:
        pdf_path (str): 输入的PDF文件的路径。
        txt_path (str): 输出的TXT文件的路径。
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''.join(reader.getPage(page_num).extract_text() for page_num in range(reader.numPages))
    except FileNotFoundError:
        print(f"File {pdf_path} not found.")
        return
    except PyPDF2.PdfReadError:
        print(f"Error reading PDF file {pdf_path}.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {pdf_path}: {e}")
        return
    with open(txt_path, 'w', encoding="utf-8") as output:
        output.write(text)

# 函数：对TXT文件进行解析
def process_txt_file(txt_path):
    """
    对TXT文件进行解析
    这一步骤是为了从TXT文件中提取有用的章节信息。
    我们首先确定目录的起始和结束位置，然后根据这些位置提取章节信息。
    
    主参数:
        txt_path (str): TXT文件的路径。
        
        函数内参数：
            document_content (str): 文档的全文内容。
            matched_pattern (re.Pattern): 检测到的章节的正则表达式模式。
            toc_start (int): “目录”的起始位置。
            toc_en_guess (int): “目录”的推测结束位置。
            content_start(int): “正文”的开始位置。    
    
    返回:
        dict: 提取的章节信息。
    """
    with open(txt_path, 'r', encoding="utf-8") as file:
        document_content = file.read()
    
    toc_start, toc_end_guess,toc_exist = determine_toc_bounds(document_content)

    try:
        matched_pattern = determine_chapter_pattern(document_content[toc_start:toc_end_guess])
        if matched_pattern is None:
            raise ValueError("根据给定的正则表达式模式，无法确定章节模式。")
    except ValueError as ve:
        print(f"Value error: {ve}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during text parsing: {e}")
        return
    
    # 根据匹配的模式提取章节信息
    chapter_matches = matched_pattern.findall(document_content[toc_start:toc_end_guess])
    chapter_numbers, chapter_titles = zip(*chapter_matches)
    
    # 将章节的中文数字转换为阿拉伯数字
    arabic_numbers = [chinese_to_arabic(cn.lstrip("第（").rstrip("章节条、）")) for cn in chapter_numbers]
    
    # 确保章节的顺序升序（出现逆序可能说明错误读取了正文中的中文数字）
    valid_indices = []
    current_max = 0
    for i, num in enumerate(arabic_numbers):
        if num > current_max:
            valid_indices.append(i)
            current_max = num

    # 获取有效的章节编号和标题
    valid_chapter_numbers = [chapter_numbers[i] for i in valid_indices]
    valid_chapter_titles = [chapter_titles[i] for i in valid_indices]
    
    #推测正文的起点（即提取章节内容的起点）
    content_start = determine_content_start(document_content, valid_chapter_titles[0], toc_start, toc_exist)

    # 提取每个章节的内容
    chapter_contents = []
    for i, chapter_number in enumerate(valid_chapter_numbers):
        start_index = find_chapter(document_content, chapter_number, content_start)
        
        # 查找章节的结束位置
        if i < len(valid_chapter_numbers) - 1:
            next_chapter_number = valid_chapter_numbers[i + 1]
            end_index = find_chapter(document_content, next_chapter_number, start_index)
        else:
            end_index = None
        
        # 获取章节内容
        chapter_content = document_content[start_index:end_index].strip()
        
        # 保存章节信息
        chapter_contents.append({
            "序号": chapter_number,
            "标题": valid_chapter_titles[i],
            "内容": chapter_content
        })
  
    return chapter_contents


import pandas as pd
from pathlib import Path
from typing import Dict, List

# 函数：将章节信息保存到Excel和CSV文件中
def save_to_excel_and_csv(results: Dict[str, List[Dict[str, str]]], folder_name: str, output_folder: Path):
    """
        这一步骤将提取的章节信息保存到Excel和CSV文件中。
    我们创建或覆盖一个Excel文件，并在其中添加多个工作表，每个工作表对应一家公司的章节信息。

    参数:
        results (Dict[str, List[Dict[str, str]]])：每家公司的章节信息字典。
        folder_name (str)：文件夹的名称。
        output_folder (Path)：输出文件夹的路径。
    """
    # 创建或覆盖Excel文件
    try:
        with pd.ExcelWriter(output_folder / f"{folder_name}.xlsx") as writer:
            for company_key, chapters in results.items():
                # 将章节信息转换为DataFrame
                df = pd.DataFrame(chapters)
                # 将DataFrame保存到Excel工作表中
                df.to_excel(writer, sheet_name=company_key, index=False)
                # 保存为CSV文件
                csv_filename = f"{folder_name}-{company_key}.csv"
                df.to_csv(output_folder / "CSV文件" / csv_filename, index=False)
    except PermissionError:
        print(f"Permission denied: Cannot write to {output_folder / f'{folder_name}.xlsx'}.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while saving Excel file: {e}")
        return    

# 函数：保存为TXT格式
def save_as_txt(results, txt_output_folder: Path):
    """
        这一步骤将提取的章节信息保存到一个TXT文件中。
    我们按照一定的格式写入每一章的信息，以便于后续的查阅或分析。

    参数:
        results (dict): 结果的字典。

    """
    txt_output_path = txt_output_folder / f"{folder_name}多文件输出结果.txt"
    with open(txt_output_path, 'w', encoding='utf-8') as file:
        file.write('='*40 + '\n')   # 写入一行由'='字符组成的分隔线
        for key, chapters in results.items():
            file.write(f"Company Key: {key}\n")   # 写入公司关键词
            file.write('-'*40 + '\n')   # 写入一行由'-'字符组成的分隔线
            for chapter in chapters:
                file.write(f"序号: {chapter['序号']}\n")   # 写入章节编号
                file.write(f"标题: {chapter['标题']}\n")   # 写入章节标题
                file.write(f"内容: {chapter['内容']}\n")   # 写入章节内容
                file.write('-'*40 + '\n')   # 写入一行由'-'字符组成的分隔线
            file.write('='*40 + '\n\n')   # 写入一行由'='字符组成的分隔线，并换行


#! 主程序开始
# 1. 使用tkinter让用户选择一个文件夹。
# 2. 列出该文件夹中的所有PDF文件。
# 3. 将PDF文件转换为TXT文件。
# 4. 对每个TXT文件进行解析，提取章节信息。
# 5. 将提取的信息保存到Excel、CSV和TXT文件中。

if __name__ == "__main__":

    # 使用tkinter让用户选择一个文件夹
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_path = Path(filedialog.askdirectory())

    # 列出目录中的所有PDF文件
    pdf_files = [f for f in folder_path.iterdir() if f.suffix == '.pdf']

    txt_files = []
    for pdf_file in pdf_files:
        txt_file = pdf_file.with_suffix('.txt')
        txt_files.append(txt_file)
        
        # 将PDF文件转换为TXT文件
        convert_pdf_to_txt(folder_path / pdf_file, folder_path / txt_file)

        # Processing
        print(f"正在将PDF转换为 {txt_file}...")

    # 定义正则表达式模式以提取公司信息
    code_pattern = re.compile(r"\d{6}\.SH|\d{6}\.SZ")
    name_pattern = re.compile(r"-((?:\w{2,3})?[\u4e00-\u9fa5]{2,4})-")

    # 处理每个TXT文件
    results = {}
    for txt_file in txt_files:
        print(f"正在处理 {txt_file}...")
        match_code = code_pattern.search(str(txt_file))
        match_name = name_pattern.search(str(txt_file))
        if not match_code or not match_name:
            print(f"文件名 {txt_file} 不符合预期的格式。跳过...")
            continue

        company_code = match_code.group()
        company_name = match_name.group(1)
        company_key = f"{company_code}-{company_name}"

        chapter_contents = process_txt_file(folder_path / txt_file)
        results[company_key] = chapter_contents

    # 提取文件夹名称
    folder_name = folder_path.name


    # 创建输出文件夹和子文件夹
    output_folder = folder_path / f"{folder_name}多文件输出结果"
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_folder = output_folder / "CSV文件"
    csv_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)    

    # Save as Excel and CSV formats
    save_to_excel_and_csv(results, folder_name, output_folder)    

    # Save as TXT format
    save_as_txt(results, output_folder)
    
    print("执行完毕！")