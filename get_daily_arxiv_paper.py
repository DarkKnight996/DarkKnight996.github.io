#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的论文处理脚本
只支持单个日期，不再支持日期段
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
import os
import re
import tempfile
from datetime import datetime, timedelta
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from bs4 import BeautifulSoup
import sys

# PDF处理相关
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("警告: PyPDF2未安装，无法处理PDF文件。请运行: pip install PyPDF2")

def already_processed(date_str, filename="arxiv_date.txt"):
    """检查 arxiv_date.txt 当前日期是否已处理过（date_str: yyyy-mm-dd）"""
    if not os.path.exists(filename):
        return False
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            yyyymmdd_list = set(line.strip() for line in lines if line.strip())
        return date_str.replace('-', '') in yyyymmdd_list
    except Exception as e:
        print(f"读取 {filename} 错误: {e}")
        return False

def append_to_processed(date_str, filename="arxiv_date.txt"):
    """处理完成后追加日期到 arxiv_date.txt（date_str: yyyy-mm-dd）"""
    try:
        with open(filename, "a") as f:
            f.write(date_str.replace('-', '') + "\n")
    except Exception as e:
        print(f"写入 {filename} 错误: {e}")

def extract_date_from_html(html_content=None, url="https://arxiv.org/list/cs/new"):
    """
    从arXiv HTML内容中提取日期
    
    Args:
        html_content (bytes or str): HTML内容，如果提供则直接使用，否则从URL下载
        url (str): arXiv HTML页面URL，仅在html_content为None时使用
        
    Returns:
        str: 日期字符串，格式为 'YYYY-MM-DD'，如果提取失败返回None
    """
    try:
        # 如果提供了HTML内容，直接使用；否则从URL下载
        if html_content is None:
            print(f"正在从 {url} 下载HTML并提取日期...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.content
        else:
            print("从提供的HTML内容中提取日期...")
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找包含"Showing new listings for"的h3标签
        h3_tags = soup.find_all('h3')
        for h3 in h3_tags:
            text = h3.get_text()
            if 'Showing new listings for' in text:
                # 提取日期部分，格式如 "Monday, 3 November 2025"
                # 匹配日期模式：Day, DD Month YYYY
                date_pattern = r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
                match = re.search(date_pattern, text)
                if match:
                    day = match.group(1)
                    month_name = match.group(2)
                    year = match.group(3)
                    
                    # 月份名称到数字的映射
                    month_map = {
                        'January': '01', 'February': '02', 'March': '03', 'April': '04',
                        'May': '05', 'June': '06', 'July': '07', 'August': '08',
                        'September': '09', 'October': '10', 'November': '11', 'December': '12'
                    }
                    
                    month_num = month_map.get(month_name)
                    if month_num:
                        # 格式化日期为 YYYY-MM-DD
                        date_str = f"{year}-{month_num}-{day.zfill(2)}"
                        print(f"从HTML页面提取到日期: {date_str}")
                        return date_str
        
        print("未能在HTML页面中找到日期信息")
        return None
        
    except Exception as e:
        print(f"从HTML页面提取日期时发生错误: {e}")
        return None

class CompletePaperProcessor:
    def __init__(self, docs_daily_path="docs/daily", temp_dir="temp_pdfs"):
        """
        初始化完整的论文处理器
        
        Args:
            docs_daily_path (str): daily文件夹路径
            temp_dir (str): 临时PDF存储目录
        """
        self.docs_daily_path = docs_daily_path
        self.temp_dir = temp_dir
        self.ensure_directories()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [self.docs_daily_path, self.temp_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    # ==================== arXiv论文获取功能 ====================

    def fetch_arxiv_papers(self, categories=['cs.DC', 'cs.AI'], max_results=2000, target_date=None, html_content=None):
        """
        从arXiv HTML内容获取指定分类的论文，并根据papers.jsonl去重与增补
        
        Args:
            categories (list): 论文分类列表（暂时忽略，从HTML获取所有cs分类）
            max_results (int): 最大获取数量
            target_date (str): 目标日期，格式为 'YYYY-MM-DD'，本函数只考虑单个日期
            html_content (bytes): HTML内容，如果提供则直接使用，否则从URL下载
            
        Returns:
            list: 论文列表（直接从HTML解析得到的论文，不再依赖papers.jsonl）
        """
        all_papers = []
        seen_papers = set()

        # 从arXiv HTML页面获取论文
        print("正在解析HTML内容获取论文信息...")
        try:
            # 如果提供了HTML内容，直接使用；否则从URL下载
            if html_content is None:
                print("正在从 https://arxiv.org/list/cs/new 下载HTML...")
                response = requests.get('https://arxiv.org/list/cs/new', timeout=30)
                response.raise_for_status()
                html_content = response.content
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找所有论文条目
            paper_entries = soup.find_all('dt')
            print(f"Found {len(paper_entries)} papers in HTML")
            
            for entry in paper_entries:
                paper_info = self._extract_paper_info_from_html(entry)
                if paper_info:
                    paper_id = paper_info.get('id', '')
                    if paper_id in seen_papers:
                        print(f"跳过重复论文: {paper_info.get('title', 'N/A')}")
                        continue
                    
                    # 检查是否是revised version
                    if paper_info.get('replaced', False):
                        # print(f"跳过revised version的论文: {paper_info.get('title', 'N/A')}")
                        continue
                    
                    # 应用筛选逻辑
                    should_add = False
                    paper_categories = paper_info.get('categories', [])
                    
                    # 先判断是否是cs.DC，是的话直接should_add，再判断cs.AI/cs.LG
                    if "cs.DC" in paper_categories:
                        should_add = True
                    elif any(cat in paper_categories for cat in categories):
                        if any(cat in ['cs.AI', 'cs.LG'] for cat in paper_categories):
                            summary_lower = paper_info.get("summary", "").lower()
                            # 标注匹配情况以便后续统计
                            paper_info['rl_match'] = "reinforcement learning" in summary_lower
                            paper_info['accelerat_match'] = "accelerat" in summary_lower
                            if paper_info['rl_match'] or paper_info['accelerat_match']:
                                should_add = True
                        else:
                            should_add = True
                    
                    if should_add:
                        all_papers.append(paper_info)
                        seen_papers.add(paper_id)
            
            print(f"成功获取 {len(all_papers)} 篇论文")
            for i, paper in enumerate(all_papers):
                print(f"{i+1}. {paper.get('title', 'N/A')}")
                
        except Exception as e:
            print(f"获取论文失败: {e}")
            return []

        print(f"总共获取 {len(all_papers)} 篇论文")

        # 不再依赖papers.jsonl，直接返回解析到的论文列表
        return all_papers
    
    def _extract_paper_info_from_html(self, dt_entry):
        """从HTML dt条目中提取论文信息"""
        try:
            # 获取对应的dd条目
            dd_entry = dt_entry.find_next_sibling('dd')
            if not dd_entry:
                print("Debug: 未找到对应的dd条目")
                return None
            
            # 提取arXiv ID和链接
            arxiv_link = dt_entry.find('a', href=lambda x: x and '/abs/' in x)
            if not arxiv_link:
                print("Debug: 未找到arXiv链接")
                return None
            
            href = arxiv_link.get('href', '')
            if href.startswith('/'):
                arxiv_id = href.split('/')[-1]
                paper_id = f"http://arxiv.org/abs/{arxiv_id}"
            else:
                arxiv_id = href.split('/')[-1]
                paper_id = href if href.startswith('http') else f"http://arxiv.org/abs/{arxiv_id}"
            
            # 检查是否有(replaced)标记
            replaced = False
            dt_text = dt_entry.get_text()
            if '(replaced)' in dt_text:
                replaced = True
            
            # 提取PDF链接
            pdf_link = "N/A"
            pdf_links = dt_entry.find_all('a', href=lambda x: x and '/pdf/' in x)
            if pdf_links:
                pdf_href = pdf_links[0].get('href', 'N/A')
                if pdf_href.startswith('/'):
                    pdf_link = f"https://arxiv.org{pdf_href}"
                else:
                    pdf_link = pdf_href
            
            # 提取标题
            title_elem = dd_entry.find('div', class_='list-title')
            title = "N/A"
            if title_elem:
                # 移除"Title:"描述符
                title_text = title_elem.get_text(strip=True)
                if title_text.startswith('Title:'):
                    title = title_text[6:].strip()
                else:
                    title = title_text
            
            # 提取作者
            authors = []
            authors_elem = dd_entry.find('div', class_='list-authors')
            if authors_elem:
                author_links = authors_elem.find_all('a')
                for author_link in author_links:
                    authors.append(author_link.get_text(strip=True))
            
            # 提取分类
            categories = []
            subjects_elem = dd_entry.find('div', class_='list-subjects')
            if subjects_elem:
                # 查找所有分类链接
                category_links = subjects_elem.find_all('a')
                for cat_link in category_links:
                    href = cat_link.get('href', '')
                    if 'searchtype=subject' in href:
                        # 从链接中提取分类代码
                        match = re.search(r'query=([^&]+)', href)
                        if match:
                            categories.append(match.group(1))
                # 如果没有找到分类链接，尝试从文本中提取
                if not categories:
                    text = subjects_elem.get_text()
                    # 匹配类似 "Machine Learning (cs.LG)" 的模式
                    matches = re.findall(r'\(([^)]+)\)', text)
                    categories = [match for match in matches if match.startswith('cs.')]
            
            # 提取摘要
            summary = "N/A"
            abstract_elem = dd_entry.find('p', class_='mathjax')
            if abstract_elem:
                summary = abstract_elem.get_text(strip=True)
            
            # 提取发布时间（从arXiv ID中推断）
            published = "N/A"
            updated = "N/A"
            if arxiv_id:
                # arXiv ID格式通常是 YYMM.NNNNN
                match = re.match(r'(\d{2})(\d{2})\.(\d+)', arxiv_id)
                if match:
                    year = "20" + match.group(1)  # 假设是20xx年
                    month = match.group(2)
                    published = f"{year}-{month}-01T00:00:00Z"
                    updated = published
            
            return {
                'id': paper_id,
                'title': title,
                'authors': authors,
                'summary': summary,
                'published': published,
                'updated': updated,
                'pdf_link': pdf_link,
                'categories': categories,
                'author_count': len(authors),
                'replaced': replaced
            }
            
        except Exception as e:
            print(f"提取论文信息时发生错误: {e}")
            return None
    
    def _extract_paper_info(self, entry, ns):
        """从XML条目中提取论文信息"""
        try:
            # 提取基本信息
            title_elem = entry.find('arxiv:title', ns)
            title = title_elem.text.strip() if title_elem is not None else "N/A"
            
            # 提取作者信息
            authors = []
            for author in entry.findall('arxiv:author', ns):
                name_elem = author.find('arxiv:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # 提取摘要
            summary_elem = entry.find('arxiv:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None else "N/A"
            
            # 提取时间信息
            published_elem = entry.find('arxiv:published', ns)
            published = published_elem.text.strip() if published_elem is not None else "N/A"
            
            updated_elem = entry.find('arxiv:updated', ns)
            updated = updated_elem.text.strip() if updated_elem is not None else "N/A"
            
            # 提取链接
            pdf_link = "N/A"
            for link in entry.findall('arxiv:link', ns):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href', "N/A")
                    break
            
            # 提取arXiv ID
            arxiv_id = entry.find('arxiv:id', ns)
            paper_id = arxiv_id.text.strip() if arxiv_id is not None else "N/A"
            
            # 提取分类
            categories = []
            for category in entry.findall('arxiv:category', ns):
                if category.get('term'):
                    categories.append(category.get('term'))
            
            return {
                'id': paper_id,
                'title': title,
                'authors': authors,
                'summary': summary,
                'published': published,
                'updated': updated,
                'pdf_link': pdf_link,
                'categories': categories,
                'author_count': len(authors),
                'replaced': False  # XML entries don't have replaced status
            }
            
        except Exception as e:
            print(f"提取论文信息时发生错误: {e}")
            return None

    def filter_by_updated_date(self, papers, date_str):
        """根据updated日期筛选论文"""
        filtered_papers = []
        for paper in papers:
            updated_field = paper.get('updated', '')
            try:
                dt = datetime.fromisoformat(updated_field.replace('Z', ''))
                if dt.strftime('%Y-%m-%d') == date_str:
                    filtered_papers.append(paper)
            except Exception:
                pass
        return filtered_papers

    # 日期段相关功能移除，不再支持
    # def filter_by_updated_date_range(self, papers, start_date, end_date):
    #     ...

    # ==================== PDF处理和LLM分析功能 ====================
    # ...无更改，省略...

    def download_pdf(self, pdf_url, filename):
        """下载PDF文件"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return filepath
        except Exception as e:
            print(f"下载PDF失败 {pdf_url}: {e}")
            return None

    def extract_first_page_text(self, pdf_path):
        """提取PDF第一页的文本内容"""
        if not PDF_AVAILABLE:
            return "PDF处理库未安装"
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    return text[:4096]  # 限制长度避免API调用过长
                else:
                    return "PDF文件为空"
        except Exception as e:
            print(f"提取PDF文本失败 {pdf_path}: {e}")
            return f"PDF处理错误: {e}"

    def call_api_for_tags_institution_interest(self, title, abstract, first_page_text):
        # ...实现保持不变...
        prompt = f"""\
Title: {title}
Abstract: {abstract}
First Page Content: {first_page_text}

Please analyze the provided paper (including its title, abstract, first page content, and author information) and generate the following structured output:

- Assign three tags:
    - tag1: Choose one of "ai", "sys", or "mlsys" based on the content. If the content is about AI algorithms, then tag1 is "ai"; if the content is about traditional system, then tag1 is "sys"; if the content is about machine learning or deep learning or AI and system, then tag1 is "mlsys".
    - tag2: If tag1 is "mlsys", select one specific subfield from the following list: "llm training", "llm inference", "multi-modal training", "multi-modal inference", "diffusion training", "diffusion inference", "post-training", "cluster infrastructure", "GPU kernels", "fault-tolerance" or "others". If tag1 is "ai" or "sys", assign any reasonable domain-specific category for tag2.
    - tag3: Provide a comma-separated list of specific methods, techniques, or keywords used in the paper (e.g., "tensor parallelism, quantization, flash attention"). For "ai" or "sys" papers, this can be any relevant technical terms.

- Identify the institution(s): Infer the main research institution(s) from author affiliations or email domains if explicit affiliations are missing.
- Finally, provide a brief llm_summary in English (2–3 sentences) describing the paper’s core method and main conclusion.

Output format (strictly follow, no extra text or code blocks):
tag1: <tag1>
tag2: <tag2>
tag3: <tag3, tag3, ...>
institution: <institution>
llm_summary: <2-3 sentences simple summary (method+conclusion)>
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. You are good at summarizing papers and extracting keywords and institutions."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            lines = [line.strip() for line in result.splitlines() if line.strip()]
            tag1, tag2, tag3, institution, llm_summary = "", "", "", "", ""
            reading_summary = False
            summary_lines = []
            
            for line in lines:
                if line.lower().startswith("tag1:"):
                    tag1 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("tag2:"):
                    tag2 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("tag3:"):
                    tag3 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("institution:"):
                    institution = line.split(":", 1)[1].strip()
                elif line.lower().startswith("llm_summary:"):
                    reading_summary = True
                    summary_line = line.split(":", 1)[1].strip()
                    if summary_line:
                        summary_lines.append(summary_line)
                elif reading_summary:
                    summary_lines.append(line)
            
            if summary_lines:
                llm_summary = ' '.join(summary_lines).strip()
            
            tag3_list = [t.strip() for t in tag3.split(',') if t.strip()]
            return tag1, tag2, tag3_list, institution, llm_summary

        except Exception as e:
            print(f"API调用失败: {e}")
            return "", "", [], "", ""

    def process_single_paper(self, paper):
        # 对于非 cs.DC 的论文，跳过PDF/LLM流程，仅用于简化输出
        categories = paper.get('categories', []) or []
        title = paper.get('title', '')
        
        if not any(cat == 'cs.DC' for cat in categories):
            paper['simple_only'] = True
            # 不再计算兴趣
            paper['is_interested'] = True
            print(f"简化处理(非cs.DC): {title}")
            return paper

        # cs.DC 才进行完整处理
        summary = paper.get('summary', '')
        pdf_link = paper.get('pdf_link', '')
        print(f"处理论文: {title}")
        
        # 下载PDF
        if not pdf_link or pdf_link == 'N/A':
            print(f"跳过论文 {title}: 无PDF链接")
            paper['is_interested'] = True
            return paper
        
        # 生成PDF文件名
        pdf_filename = f"{paper.get('id', '').split('/')[-1]}.pdf"
        
        # 下载PDF
        pdf_path = self.download_pdf(pdf_link, pdf_filename)
        if not pdf_path:
            print(f"跳过论文 {title}: PDF下载失败")
            paper['is_interested'] = True
            return paper
        
        # 提取第一页文本
        first_page_text = self.extract_first_page_text(pdf_path)
        
        # 调用API获取标签、机构，并获取LLM总结
        tag1, tag2, tag3_list, institution, llm_summary = self.call_api_for_tags_institution_interest(
            title, summary, first_page_text
        )
        
        # 更新论文信息
        paper['tag1'] = tag1
        paper['tag2'] = tag2
        paper['tag3'] = ', '.join(tag3_list)
        paper['institution'] = institution
        # 所有 cs.DC 都输出
        paper['is_interested'] = True
        paper['llm_summary'] = llm_summary
        paper['simple_only'] = False
        
        # 清理临时PDF文件
        try:
            os.remove(pdf_path)
        except:
            pass
        
        print(f"完成论文 {title}: tag1={tag1}, tag2={tag2}, institution={institution}")
        return paper
    
    # ==================== Markdown文件处理功能 ====================
    # ...实现不变，省略...
    def get_week_range(self, date_str):
        """根据日期获取该周的周一到周日的日期范围"""
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_since_monday = target_date.weekday()
            monday = target_date - timedelta(days=days_since_monday)
            sunday = monday + timedelta(days=6)
            
            start_str = monday.strftime('%Y%m%d')
            end_str = sunday.strftime('%Y%m%d')
            
            return f"{start_str}-{end_str}"
        except ValueError as e:
            print(f"日期格式错误: {e}")
            return None
    
    def get_arxiv_prefix(self, date_str):
        """根据日期获取类似[arXiv251027]的字符串"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            prefix = f"[arXiv{str(dt.year)[-2:]}{dt.month:02d}{dt.day:02d}]"
            return prefix
        except Exception:
            return ""

    def format_paper_with_enhanced_info(self, paper, date_str=None):
        # 非 cs.DC 使用简化格式：- [arXivYYMMDD] title [link](https://...)
        categories = paper.get('categories', []) or []
        title = paper.get('title', 'N/A')
        arxiv_prefix = ""
        if date_str is not None:
            arxiv_prefix = self.get_arxiv_prefix(date_str)
        else:
            arxiv_prefix = ""
        if not any(cat == 'cs.DC' for cat in categories):
            pdf_link = paper.get('pdf_link', '')
            paper_link = pdf_link if pdf_link and pdf_link != 'N/A' else paper.get('id', '')
            return f"- {arxiv_prefix} {title} [link]({paper_link})\n"

        # cs.DC 使用详细格式
        authors = ', '.join(paper.get('authors', []))
        pdf_link = paper.get('pdf_link', 'N/A')
        
        tags = []
        if paper.get('tag1'):
            tags.append("[" + paper['tag1'] + "]")
        if paper.get('tag2'):
            tags.append("[" + paper['tag2'] + "]")
        if paper.get('tag3'):
            tag3_items = [t.strip() for t in paper['tag3'].split(',') if t.strip()]
            if tag3_items:
                tags.append('[' + ', '.join(tag3_items) + ']')
        tags_str = ', '.join(tags) if tags else 'TBD'
        institution = paper.get('institution', 'TBD')
        llm_summary = paper.get('llm_summary', '').strip()
        
        formatted_text = f"""- **{arxiv_prefix} {title}**
  - **tags:** {tags_str}
  - **authors:** {authors}
  - **institution:** {institution}
  - **link:** {pdf_link}
"""
        if llm_summary:
            # 转义MDX特殊字符：大括号{}会被MDX解析为JSX表达式，需要转义
            escaped_summary = llm_summary.replace('<', '&lt;').replace('>', '&gt;').replace('{', '\\{').replace('}', '\\}')
            formatted_text += f"  - **Simple LLM Summary:** {escaped_summary}\n"
        formatted_text += "\n"
        return formatted_text

    def update_markdown_file(self, filepath, papers, date_str):
        # ...实现不变...
        if not papers:
            print("没有论文需要添加")
            return

        # 不再根据兴趣过滤，全部输出
        all_papers = papers

        existing_content = ""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_content = f.read()

        # 利用正则找到所有日期section
        date_section_pattern = re.compile(
            r"(^|\n)##\s*(\d{4}-\d{2}-\d{2}).*?(?=\n##\s|\Z)", re.DOTALL
        )
        all_sections = []
        for m in date_section_pattern.finditer(existing_content):
            section_start = m.start()
            section_content = m.group(0).lstrip('\n')
            section_date = m.group(2)
            all_sections.append((section_date, section_content, section_start))
        
        # 新section内容
        papers_content = f"## {date_str}\n\n"
        if all_papers:
            # 先输出 cs.DC，再输出其他，保持各自相对顺序，并在每类开头输出总数
            csdc_papers = [p for p in all_papers if any(cat == 'cs.DC' for cat in (p.get('categories', []) or []))]
            other_papers = [p for p in all_papers if not any(cat == 'cs.DC' for cat in (p.get('categories', []) or []))]
            # 统计 cs.AI/cs.LG 两组关键词
            rl_papers = [p for p in other_papers if p.get('rl_match')]
            accelerat_papers = [p for p in other_papers if p.get('accelerat_match')]

            papers_content += f"**cs.DC total: {len(csdc_papers)}**\n\n"
            for paper in csdc_papers:
                papers_content += self.format_paper_with_enhanced_info(paper, date_str=date_str)

            papers_content += f"\n**cs.AI/cs.LG contains \"reinforcement learning\" total: {len(rl_papers)}**\n"
            for paper in rl_papers:
                papers_content += self.format_paper_with_enhanced_info(paper, date_str=date_str)

            papers_content += f"\n**cs.AI/cs.LG contains \"accelerate\" total: {len(accelerat_papers)}**\n"
            for paper in accelerat_papers:
                papers_content += self.format_paper_with_enhanced_info(paper, date_str=date_str)
        else:
            papers_content += "No papers today\n"

        replaced = False
        # 如有则替换当前日期section
        for idx, (dt, _, start_idx) in enumerate(all_sections):
            if dt == date_str:
                # 替换
                before = existing_content[:start_idx].rstrip('\n')
                after_idx = start_idx + len(_)
                after = existing_content[after_idx:]
                new_content = before
                if new_content and not new_content.endswith('\n'):
                    new_content += "\n"
                new_content += "\n" + papers_content
                if after and not after.startswith('\n'):
                    new_content += "\n"
                new_content += after.lstrip('\n')
                replaced = True
                print(f"日期 {date_str} 的内容已存在，已覆盖")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content.strip() + '\n')
                print(f"已将 {len(all_papers)} 篇论文添加到文件: {filepath}")
                return

        # 如果没有，插入保持时间递增顺序（从小到大）
        # 找到插入点：第一个section日期大于本date_str，则插入在它前面；若找不到，追加到文件末尾
        insert_idx = None
        for idx, (dt, _, start_idx) in enumerate(all_sections):
            if dt > date_str:
                insert_idx = start_idx
                break
        if insert_idx is not None:
            # 插入到insert_idx前
            before = existing_content[:insert_idx].rstrip('\n')
            after = existing_content[insert_idx:]
            new_content = before
            if new_content and not new_content.endswith('\n'):
                new_content += "\n"
            new_content += "\n" + papers_content
            if after and not after.startswith('\n'):
                new_content += "\n"
            new_content += after.lstrip('\n')
            print(f"日期 {date_str} 的内容不存在，已按时间顺序插入")
        else:
            # 追加到最后
            new_content = existing_content.rstrip() + "\n\n" + papers_content
            print(f"日期 {date_str} 的内容不存在，已追加到最后")
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content.strip() + '\n')

        print(f"已将 {len(all_papers)} 篇论文添加到文件: {filepath}")

    def find_or_create_weekly_file(self, date_str):
        """根据日期找到或创建对应的周文件"""
        week_range = self.get_week_range(date_str)
        if not week_range:
            return None
        
        filename = f"{week_range}.md"
        filepath = os.path.join(self.docs_daily_path, filename)
        
        if not os.path.exists(filepath):
            self.create_weekly_file(filepath, week_range)
        
        return filepath

    def create_weekly_file(self, filepath, week_range):
        """创建新的周文件"""
        start_date_str, end_date_str = week_range.split('-')
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.strptime(end_date_str, '%Y%m%d')
        
        content = f"""# {week_range}

"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"创建新的周文件: {filepath}")

    # ==================== 主处理流程 ====================
    
    def process_papers_by_date(self, target_date=None, categories=['cs.DC', 'cs.AI'], max_workers=2, max_papers=10, html_content=None):
        """
        根据指定日期处理论文的完整流程

        Args:
            target_date (str): 目标日期，格式为 'YYYY-MM-DD'
            categories (list): 论文分类列表
            max_workers (int): 并发处理数量
            max_papers (int): 最大处理论文数量（用于测试）
            html_content (bytes): HTML内容，如果提供则直接使用
        """
        # 若未提供日期，则默认使用今天
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')

        # ==== 新增: arxiv_date.txt 检查 ====
        today_ymd = target_date
        if already_processed(today_ymd):
            print(f"日期 {today_ymd} 已经处理过，自动退出。")
            return

        print(f"开始处理日期: {target_date}")

        single_date = target_date
        print(f"\n==== 处理 {single_date} ====")
        # 1. 从arXiv获取论文
        print("步骤1: 从arXiv获取论文...")
        papers = self.fetch_arxiv_papers(categories=categories, max_results=1024, target_date=single_date, html_content=html_content)

        if not papers:
            print(f"日期 {single_date} 没有找到论文")
            append_to_processed(single_date)
            return

        # 限制处理数量（用于测试）
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            print(f"限制处理前 {max_papers} 篇论文")

        print(f"找到 {len(papers)} 篇论文，开始处理...")

        # 2. 并发处理论文（下载PDF、调用LLM）
        print("步骤2: 处理论文（下载PDF、调用LLM）...")
        processed_papers = []

        for i, paper in enumerate(papers):
            print(f"{i+1}. {paper.get('title', 'N/A')}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.process_single_paper, paper): paper 
                for paper in papers
            }

            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_paper), 
                             total=len(future_to_paper), desc="处理论文"):
                try:
                    processed_paper = future.result()
                    processed_papers.append(processed_paper)
                except Exception as e:
                    print(f"处理论文时出错: {e}")

        # 3. 统计结果
        print(f"处理完成！总共 {len(processed_papers)} 篇论文")

        # 4. 更新markdown文件
        print("步骤3: 更新markdown文件...")
        weekly_file = self.find_or_create_weekly_file(single_date)
        if weekly_file:
            self.update_markdown_file(weekly_file, processed_papers, single_date)
            print(f"处理完成！论文已添加到: {weekly_file}")
        else:
            print("无法创建或找到周文件")
        
        # 完成后写入arxiv_date.txt
        append_to_processed(single_date)

def main():
    """
    主函数 - 使用示例
    """
    if not PDF_AVAILABLE:
        print("请先安装PyPDF2: pip install PyPDF2")
        return
    
    # 检查API密钥
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    # 从arXiv HTML页面下载HTML内容（只下载一次）
    arxiv_url = "https://arxiv.org/list/cs/new"
    print(f"正在从 {arxiv_url} 下载HTML内容...")
    try:
        response = requests.get(arxiv_url, timeout=30)
        response.raise_for_status()
        html_content = response.content
        print("HTML内容下载成功")
    except Exception as e:
        print(f"下载HTML内容失败: {e}")
        html_content = None
    
    # 从HTML内容中提取日期
    if html_content:
        target_date = extract_date_from_html(html_content=html_content)
    else:
        target_date = None
    
    # 如果从HTML页面提取失败，使用当前日期
    if not target_date:
        print("无法从HTML页面提取日期，使用当前日期")
        target_date = datetime.now().strftime('%Y-%m-%d')
    else:
        print(f"使用从HTML页面提取的日期: {target_date}")

    # ==== 运行前检查日期是否已处理 ====
    if already_processed(target_date):
        print(f"日期 {target_date} 已经处理过，自动退出。")
        return

    # 创建处理器
    processor = CompletePaperProcessor()
    
    # 处理论文（传递已下载的HTML内容）
    processor.process_papers_by_date(
        target_date=target_date,
        categories=['cs.DC', 'cs.AI', 'cs.LG'],  # 可以修改分类
        max_workers=10,  # 并发数量，建议不要太高
        max_papers=None,    # 测试时限制论文数量，正式使用时可以设为None
        html_content=html_content  # 传递已下载的HTML内容
    )

if __name__ == "__main__":
    main()